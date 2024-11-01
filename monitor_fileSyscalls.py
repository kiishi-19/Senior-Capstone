from bcc import BPF

# BPF program
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
// Your BPF code here...
"""

# Load BPF program
b = BPF(text=bpf_text)

# Setup file to write output
output_file = open("syscall_output.txt", "w")

try:
    print("Tracing... Press Ctrl-C to end.")
    while True:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        output = f"{ts:.9f} {task.decode()} {pid} {msg.decode()}\n"
        output_file.write(output)
        output_file.flush()
except KeyboardInterrupt:
    print("Detaching...")
    output_file.close()


