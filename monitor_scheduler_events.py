from bcc import BPF

# Simplified BPF program monitoring `sched_process_exec`, `sched_process_fork`, and `sched_process_exit`
bpf_text = """
TRACEPOINT_PROBE(sched, sched_process_exec) {
    bpf_trace_printk("Exec: PID %d\\n", args->pid);
    return 0;
}

TRACEPOINT_PROBE(sched, sched_process_fork) {
    bpf_trace_printk("Fork: Parent PID %d, Child PID %d\\n", args->parent_pid, args->child_pid);
    return 0;
}

TRACEPOINT_PROBE(sched, sched_process_exit) {
    bpf_trace_printk("Exit: PID %d\\n", args->pid);
    return 0;
}
"""

# Load BPF program
b = BPF(text=bpf_text)

print("Monitoring scheduler events (exec, fork, exit)... Press Ctrl-C to stop.")
try:
    while True:
        (_, _, _, _, _, msg) = b.trace_fields()
        print(msg)
except KeyboardInterrupt:
    print("Detaching...")
