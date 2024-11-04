import tkinter as tk  # Tkinter is used for creating the graphical user interface (GUI)
from tkinter import ttk  # ttk provides themed widgets for a more modern look in Tkinter
import psutil  # psutil library is used to retrieve system metrics
import threading  # threading allows the monitoring process to run in the background
import time  # time is used to control the update frequency of the metrics

class SystemMonitorApp:
    def __init__(self, root):
        # Initialize the main window with a title and background color
        self.root = root
        self.root.title("CyberDawgs - Incident Management")  # Set the title of the window
        self.root.geometry("500x400")  # Set the initial size of the window
        self.root.configure(bg="#2d2d2d")  # Dark gray background for a modern look

        # Define fonts and colors for the text and labels
        self.title_font = ("Arial", 16, "bold")  # Font for the main title
        self.label_font = ("Arial", 12, "bold")  # Font for metric labels
        self.text_color = "#ffffff"  # White color for text to improve visibility
        self.highlight_color = "#4caf50"  # Light green color to highlight high metric values

        # Flag to control the monitoring thread (True for active, False for stop)
        self.monitoring = False

        # Create the title at the top left
        self.create_title()
        
        # Create UI elements like labels and buttons
        self.create_widgets()
    
    def create_title(self):
        # Title label positioned at the top left corner
        title_label = tk.Label(self.root, text="CyberDawgs - Incident Management", font=self.title_font, fg=self.text_color, bg="#2d2d2d")
        title_label.place(x=10, y=10)  # Place the title at coordinates (10,10)

    def create_widgets(self):
        # Frame to contain metric labels, positioned under the title
        self.label_frame = tk.Frame(self.root, bg="#2d2d2d")  # Create a frame for the metrics
        self.label_frame.pack(expand=True, pady=(50, 0))  # Center the frame and add vertical padding

        # CPU Usage label
        self.cpu_label = ttk.Label(self.label_frame, text="CPU Usage: ", font=self.label_font, foreground=self.text_color, background="#2d2d2d")
        self.cpu_label.pack(pady=5)  # Add vertical padding for spacing
        
        # Memory Usage label
        self.memory_label = ttk.Label(self.label_frame, text="Memory Usage: ", font=self.label_font, foreground=self.text_color, background="#2d2d2d")
        self.memory_label.pack(pady=5)
        
        # Disk Usage label
        self.disk_label = ttk.Label(self.label_frame, text="Disk Usage: ", font=self.label_font, foreground=self.text_color, background="#2d2d2d")
        self.disk_label.pack(pady=5)
        
        # Network Bytes label
        self.network_label = ttk.Label(self.label_frame, text="Network (Sent/Receive): ", font=self.label_font, foreground=self.text_color, background="#2d2d2d")
        self.network_label.pack(pady=5)
        
        # Active Connections label
        self.connection_label = ttk.Label(self.label_frame, text="Active Connections: ", font=self.label_font, foreground=self.text_color, background="#2d2d2d")
        self.connection_label.pack(pady=5)

        # Start Monitoring button with green background
        self.start_button = tk.Button(self.root, text="Start Monitoring", command=self.start_monitoring, font=("Arial", 10), bg="#4caf50", fg="white", padx=10, pady=5)
        self.start_button.pack(side="left", padx=20, pady=20)  # Position the button on the left with padding
        
        # Stop Monitoring button with red background
        self.stop_button = tk.Button(self.root, text="Stop Monitoring", command=self.stop_monitoring, font=("Arial", 10), bg="#f44336", fg="white", padx=10, pady=5)
        self.stop_button.pack(side="right", padx=20, pady=20)  # Position the button on the right with padding
    
    def start_monitoring(self):
        # Start the monitoring process in a separate thread to prevent UI freezing
        if not self.monitoring:
            self.monitoring = True  # Set flag to True to indicate active monitoring
            # Start a new thread to continuously update metrics
            self.monitor_thread = threading.Thread(target=self.update_metrics)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        # Stop the monitoring process by setting the flag to False
        self.monitoring = False
    
    def update_metrics(self):
        # Continuously update metrics while monitoring is active
        while self.monitoring:
            # Retrieve CPU usage percentage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Retrieve memory usage percentage
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            # Retrieve disk usage percentage
            disk_usage = psutil.disk_usage('/').percent
            
            # Retrieve network I/O statistics (sent and received bytes)
            net_io = psutil.net_io_counters()
            network_usage = f"{net_io.bytes_sent / (1024**2):.2f} MB / {net_io.bytes_recv / (1024**2):.2f} MB"  # Convert bytes to MB
            
            # Count active network connections in 'ESTABLISHED' state
            connections = psutil.net_connections()
            active_connections = sum(1 for conn in connections if conn.status == 'ESTABLISHED')

            # Update labels with the metrics and apply highlight color if usage exceeds 50%
            self.cpu_label.config(text=f"CPU Usage: {cpu_usage}%", foreground=self.highlight_color if cpu_usage > 50 else self.text_color)
            self.memory_label.config(text=f"Memory Usage: {memory_usage}%", foreground=self.highlight_color if memory_usage > 50 else self.text_color)
            self.disk_label.config(text=f"Disk Usage: {disk_usage}%", foreground=self.highlight_color if disk_usage > 50 else self.text_color)
            self.network_label.config(text=f"Network (Sent/Receive): {network_usage}")
            self.connection_label.config(text=f"Active Connections: {active_connections}")
            
            # Delay for the next update to control the refresh rate
            time.sleep(1)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()  # Create the main Tkinter window
    app = SystemMonitorApp(root)  # Instantiate the SystemMonitorApp class
    root.mainloop()  # Start the Tkinter event loop to display the window
