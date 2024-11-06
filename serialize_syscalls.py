import sys
from syscall_event_pb2 import SyscallEvent, ProbeType

# Map probe names from your BPFtrace script to the enum values
probe_map = {
    "sys_enter_openat": ProbeType.SYS_ENTER_OPENAT,
    "sys_enter_read": ProbeType.SYS_ENTER_READ,
    "sys_enter_write": ProbeType.SYS_ENTER_WRITE,
    "sys_enter_unlink": ProbeType.SYS_ENTER_UNLINK,
    "sys_enter_chmod": ProbeType.SYS_ENTER_CHMOD,
    "sys_enter_chown": ProbeType.SYS_ENTER_CHOWN,
    "sys_enter_execve": ProbeType.SYS_ENTER_EXECVE,
    "sys_enter_fork": ProbeType.SYS_ENTER_FORK,
    "sys_enter_clone": ProbeType.SYS_ENTER_CLONE,
    "sys_enter_kill": ProbeType.SYS_ENTER_KILL,
    "sys_enter_exit_group": ProbeType.SYS_ENTER_EXIT_GROUP,
    "sys_enter_socket": ProbeType.SYS_ENTER_SOCKET,
    "sys_enter_connect": ProbeType.SYS_ENTER_CONNECT,
    "sys_enter_bind": ProbeType.SYS_ENTER_BIND,
    "sys_enter_listen": ProbeType.SYS_ENTER_LISTEN,
    "sys_enter_accept": ProbeType.SYS_ENTER_ACCEPT,
    "sys_enter_getuid": ProbeType.SYS_ENTER_GETUID,
    "sys_enter_setuid": ProbeType.SYS_ENTER_SETUID,
    "sys_enter_sethostname": ProbeType.SYS_ENTER_SETHOSTNAME,
    "sys_enter_mount": ProbeType.SYS_ENTER_MOUNT,
    "sys_enter_umount": ProbeType.SYS_ENTER_UMOUNT,
    "sched_process_fork": ProbeType.SCHED_PROCESS_FORK,
    "sched_process_exec": ProbeType.SCHED_PROCESS_EXEC,
    "sched_process_exit": ProbeType.SCHED_PROCESS_EXIT,
    "module_load": ProbeType.MODULE_LOAD,
    "module_free": ProbeType.MODULE_FREE,
    "cpu_frequency": ProbeType.CPU_FREQUENCY
}

def main():
    with open("syscall_log.bin", "ab") as bin_file:
        for line in sys.stdin:
            # Skip lines that do not start with a numeric timestamp
            if not line.strip().split('|')[0].isdigit():
                continue

            try:
                parts = line.strip().split('|')
                event = SyscallEvent(
                    timestamp=int(parts[0]),
                    probe=probe_map.get(parts[1], ProbeType.UNKNOWN),
                    pid=int(parts[2]),
                    comm=parts[3]
                )
                bin_file.write(event.SerializeToString())
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line}, {e}")

if __name__ == "__main__":
    main()
