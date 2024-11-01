from bcc import BPF

# BPF program
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// Trace clone syscall
int kprobe__sys_clone(struct pt_regs *ctx, unsigned long flags) {
    bpf_trace_printk("clone called: PID %d, flags: %lx\\n", bpf_get_current_pid_tgid() >> 32, flags);
    return 0;
}

// Add additional probes here as necessary...
"""

# Load BPF program
b = BPF(text=bpf_text)
b.attach_kprobe(event="__x64_sys_clone", fn_name="kprobe__sys_clone")

print("Tracing syscalls... Press Ctrl-C to end.")
try:
    while True:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        print("%-18.9f %-16s %-6d %s" % (ts, task, pid, msg))
except KeyboardInterrupt:
    print("Detaching...")
    exit()
