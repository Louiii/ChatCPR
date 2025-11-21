#!/usr/bin/env python3
"""
Multi-panel log file viewer that monitors output*.log files
and displays them in vertical panels with live updates.
"""

import curses
import glob
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class LogMonitor:
    def __init__(self, pattern="output*.log"):
        self.pattern = pattern
        self.log_files = {}
        self.file_positions = {}
        self.last_update_times = {}
        self.log_contents = defaultdict(list)
        
    def discover_files(self):
        """Find all files matching the pattern"""
        current_files = set(glob.glob(self.pattern))
        
        # Add new files
        for filepath in current_files:
            if filepath not in self.log_files:
                self.log_files[filepath] = open(filepath, 'r')
                self.file_positions[filepath] = 0
                self.last_update_times[filepath] = time.time()
                # Read existing content
                self.log_files[filepath].seek(0, 2)  # Go to end
                self.file_positions[filepath] = self.log_files[filepath].tell()
        
        # Remove files that no longer exist
        for filepath in list(self.log_files.keys()):
            if filepath not in current_files:
                self.log_files[filepath].close()
                del self.log_files[filepath]
                del self.file_positions[filepath]
                del self.last_update_times[filepath]
                del self.log_contents[filepath]
    
    def extract_name(self, filepath):
        """Extract the * part from output*.log"""
        basename = os.path.basename(filepath)
        if basename.startswith('output') and basename.endswith('.log'):
            return basename[6:-4]  # Remove 'output' and '.log'
        return basename
    
    def read_updates(self):
        """Read new content from all monitored files"""
        files_to_remove = []
        
        for filepath, file_obj in list(self.log_files.items()):
            try:
                # Check if file still exists
                if not os.path.exists(filepath):
                    files_to_remove.append(filepath)
                    continue
                
                # Check if file has new content
                current_pos = file_obj.tell()
                file_obj.seek(0, 2)  # Go to end
                end_pos = file_obj.tell()
                
                if end_pos > current_pos:
                    file_obj.seek(current_pos)
                    new_content = file_obj.read()
                    if new_content:
                        lines = new_content.split('\n')
                        # Filter out empty lines at the end
                        lines = [line for line in lines if line.strip()]
                        self.log_contents[filepath].extend(lines)
                        self.last_update_times[filepath] = time.time()
                        self.file_positions[filepath] = file_obj.tell()
            except Exception as e:
                # File might have been rotated or deleted
                files_to_remove.append(filepath)
        
        # Clean up files that no longer exist
        for filepath in files_to_remove:
            try:
                self.log_files[filepath].close()
            except:
                pass
            del self.log_files[filepath]
            del self.file_positions[filepath]
            del self.last_update_times[filepath]
            # Keep the log contents so they remain visible
            # del self.log_contents[filepath]
    
    def get_time_since_update(self, filepath):
        """Get human-readable time since last update"""
        if filepath not in self.last_update_times:
            return "n/a"
        
        elapsed = time.time() - self.last_update_times[filepath]
        
        if elapsed < 1:
            return "just now"
        elif elapsed < 60:
            return f"{int(elapsed)}s ago"
        elif elapsed < 3600:
            return f"{int(elapsed/60)}m ago"
        else:
            return f"{int(elapsed/3600)}h ago"


def draw_panels(stdscr, monitor):
    """Draw the terminal UI with vertical panels"""
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)   # Non-blocking input
    stdscr.timeout(100) # Refresh every 100ms
    
    # Colors
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
    
    while True:
        # Check for 'q' to quit
        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            break
        
        # Discover new files and read updates
        monitor.discover_files()
        monitor.read_updates()
        
        # Get terminal size
        height, width = stdscr.getmaxyx()
        
        # Get all files (including active and deleted but still visible)
        all_files = sorted(set(list(monitor.log_files.keys()) + list(monitor.log_contents.keys())))
        num_files = len(all_files)
        
        if num_files == 0:
            stdscr.clear()
            stdscr.addstr(height // 2, width // 2 - 20, 
                         "No log files found. Waiting...", 
                         curses.color_pair(2))
            stdscr.refresh()
            time.sleep(0.5)
            continue
        
        # Calculate panel dimensions with multi-row support
        # If height > 2 * width for a single column layout, use multiple rows
        single_col_width = width // num_files
        single_col_height = height - 2  # Reserve 2 lines for header
        
        # Determine if we need multiple rows
        if single_col_height > 2 * single_col_width and num_files > 1:
            # Use multiple rows
            cols_per_row = max(1, int((num_files + 1) // 2))  # Aim for 2 rows
            num_rows = (num_files + cols_per_row - 1) // cols_per_row
            panel_width = width // cols_per_row
            panel_height = height // num_rows
        else:
            # Use single row
            cols_per_row = num_files
            num_rows = 1
            panel_width = width // num_files
            panel_height = height
        
        stdscr.clear()
        
        # Draw panels
        for idx, filepath in enumerate(all_files):
            row = idx // cols_per_row
            col = idx % cols_per_row
            
            x_start = col * panel_width
            y_start = row * panel_height
            
            content_height = panel_height - 2  # Reserve 2 lines for header
            
            # Check if file still exists
            file_exists = filepath in monitor.log_files
            
            # Draw header
            name = monitor.extract_name(filepath)
            time_info = monitor.get_time_since_update(filepath)
            header = f" {name} "
            
            if not file_exists:
                time_str = f" [MOVED/DELETED] "
                header_color = curses.color_pair(4)  # Red for deleted
                time_color = curses.color_pair(4)
            else:
                time_str = f" {time_info} "
                header_color = curses.color_pair(1) | curses.A_BOLD
                time_color = curses.color_pair(3)
            
            # Truncate if too long
            max_header_len = panel_width - 2
            if len(header) > max_header_len:
                header = header[:max_header_len-3] + "..."
            
            try:
                stdscr.addstr(y_start, x_start, header.ljust(panel_width)[:panel_width], 
                            header_color)
                stdscr.addstr(y_start + 1, x_start, time_str.ljust(panel_width)[:panel_width], 
                            time_color)
            except curses.error:
                pass
            
            # Draw vertical separator (right side of panel)
            if col < cols_per_row - 1:
                for y in range(y_start, min(y_start + panel_height, height)):
                    try:
                        stdscr.addch(y, x_start + panel_width - 1, '│')
                    except curses.error:
                        pass
            
            # Draw horizontal separator (bottom of panel)
            if row < num_rows - 1:
                for x in range(x_start, min(x_start + panel_width, width)):
                    try:
                        stdscr.addch(y_start + panel_height - 1, x, '─')
                    except curses.error:
                        pass
            
            # Draw log content (scrolling from bottom, padded if needed)
            log_lines = monitor.log_contents.get(filepath, [])
            
            # Take last N lines that fit in the panel
            num_lines_needed = content_height - 1
            if len(log_lines) < num_lines_needed:
                # Pad with empty lines at the beginning
                visible_lines = [''] * (num_lines_needed - len(log_lines)) + log_lines
            else:
                visible_lines = log_lines[-num_lines_needed:]
            
            for line_idx, line in enumerate(visible_lines):
                y_pos = y_start + 2 + line_idx
                if y_pos >= y_start + panel_height or y_pos >= height:
                    break
                
                # Truncate line to fit panel width
                display_line = line[:panel_width-2]
                try:
                    stdscr.addstr(y_pos, x_start + 1, display_line)
                except curses.error:
                    pass
        
        # Draw instructions at the bottom
        try:
            instruction = "Press 'q' to quit"
            stdscr.addstr(height - 1, width - len(instruction) - 1, 
                         instruction, curses.color_pair(2))
        except curses.error:
            pass
        
        stdscr.refresh()
        time.sleep(0.1)


def main():
    # Get pattern from command line or use default
    pattern = sys.argv[1] if len(sys.argv) > 1 else "output*.log"
    
    monitor = LogMonitor(pattern)
    
    try:
        curses.wrapper(lambda stdscr: draw_panels(stdscr, monitor))
    except KeyboardInterrupt:
        pass
    finally:
        # Close all open files
        for file_obj in monitor.log_files.values():
            file_obj.close()
    
    print("\nLog viewer closed.")


if __name__ == "__main__":
    main()
