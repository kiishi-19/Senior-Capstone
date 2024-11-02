from bcc import BPF

# BPF program that attaches to the incoming TCP connection handling function
bpf_text = """
#include <net/inet_sock.h>
#include <bcc/proto.h>

int kprobe__tcp_v4_do_rcv(struct pt_regs *ctx, struct sock *sk) {
    struct inet_sock *inet = (struct inet_sock *)sk;
    if (inet != NULL) {
        u32 saddr = inet->inet_saddr;
        u32 daddr = inet->inet_daddr;
        u16 sport = bpf_ntohs(inet->inet_sport);
        u16 dport = bpf_ntohs(inet->inet_dport);

        // Split the printing into multiple calls
        bpf_trace_printk("Received TCP:\\n");
        bpf_trace_printk("SRC IP: %x, DST IP: %x\\n", saddr, daddr);
        bpf_trace_printk("SRC Port: %d, DST Port: %d\\n", sport, dport);
    }
    return 0;
}
"""

# Load BPF program
b = BPF(text=bpf_text)
b.attach_kprobe(event="tcp_v4_do_rcv", fn_name="kprobe__tcp_v4_do_rcv")

print("Monitoring incoming TCP connections... Press Ctrl-C to end.")
try:
    while True:
        (_, _, _, _, _, msg) = b.trace_fields()
        print(msg)
except KeyboardInterrupt:
    print("Detaching...")
    exit()

