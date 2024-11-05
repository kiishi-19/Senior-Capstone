#!/bin/bash

# Logging output file
OUTPUT_LOG="tracer_output.log"
# Start tracer.bt in background
bpftrace tracer.bt > $OUTPUT_LOG 2>&1 & 
BPFTRACE_PID=$!

# Give bpftrace some time to start up
sleep 2

# Test File System Events
touch temp.txt
cat temp.txt
echo "Hello" > temp.txt
rm temp.txt
touch temp.txt
chmod 777 temp.txt
chown nobody:nogroup temp.txt 2>/dev/null # This might fail if not root

# Test Process Control Events
ls >/dev/null
bash -c 'sleep 1' & # Fork and clone
kill -9 $$ & # This will kill the script runner process; may want to use a subshell or another target
sleep 1 # Allow background processes to execute

# Test Network Operations
python3 -c 'import socket; s=socket.socket(); s.close()' 2>/dev/null
nc -zv localhost 22 2>/dev/null
(sleep 10 & nc -l 9999 >/dev/null 2>&1 & sleep 1; nc localhost 9999 >/dev/null 2>&1) &

# Test Security-Sensitive Syscalls
id -u >/dev/null
sudo hostname tempname 2>/dev/null
sudo mount --bind /tmp /mnt/tmp 2>/dev/null
sudo umount /mnt/tmp 2>/dev/null

# Test Kernel Module Events
sudo modprobe dummy 2>/dev/null
sudo modprobe -r dummy 2>/dev/null

# Test CPU Frequency Events (if applicable)
echo performance | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null

# Cleanup
kill $BPFTRACE_PID
wait $BPFTRACE_PID 2>/dev/null

# Display the log output
echo "Test complete. Log output:"
cat $OUTPUT_LOG
