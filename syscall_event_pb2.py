# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: syscall_event.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='syscall_event.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x13syscall_event.proto\"W\n\x0cSyscallEvent\x12\x11\n\ttimestamp\x18\x01 \x01(\x04\x12\x19\n\x05probe\x18\x02 \x01(\x0e\x32\n.ProbeType\x12\x0b\n\x03pid\x18\x03 \x01(\x05\x12\x0c\n\x04\x63omm\x18\x04 \x01(\t*\xe0\x04\n\tProbeType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x14\n\x10SYS_ENTER_OPENAT\x10\x01\x12\x12\n\x0eSYS_ENTER_READ\x10\x02\x12\x13\n\x0fSYS_ENTER_WRITE\x10\x03\x12\x14\n\x10SYS_ENTER_UNLINK\x10\x04\x12\x13\n\x0fSYS_ENTER_CHMOD\x10\x05\x12\x13\n\x0fSYS_ENTER_CHOWN\x10\x06\x12\x14\n\x10SYS_ENTER_EXECVE\x10\x07\x12\x12\n\x0eSYS_ENTER_FORK\x10\x08\x12\x13\n\x0fSYS_ENTER_CLONE\x10\t\x12\x12\n\x0eSYS_ENTER_KILL\x10\n\x12\x18\n\x14SYS_ENTER_EXIT_GROUP\x10\x0b\x12\x14\n\x10SYS_ENTER_SOCKET\x10\x0c\x12\x15\n\x11SYS_ENTER_CONNECT\x10\r\x12\x12\n\x0eSYS_ENTER_BIND\x10\x0e\x12\x14\n\x10SYS_ENTER_LISTEN\x10\x0f\x12\x14\n\x10SYS_ENTER_ACCEPT\x10\x10\x12\x14\n\x10SYS_ENTER_GETUID\x10\x11\x12\x14\n\x10SYS_ENTER_SETUID\x10\x12\x12\x19\n\x15SYS_ENTER_SETHOSTNAME\x10\x13\x12\x13\n\x0fSYS_ENTER_MOUNT\x10\x14\x12\x14\n\x10SYS_ENTER_UMOUNT\x10\x15\x12\x16\n\x12SCHED_PROCESS_FORK\x10\x16\x12\x16\n\x12SCHED_PROCESS_EXEC\x10\x17\x12\x16\n\x12SCHED_PROCESS_EXIT\x10\x18\x12\x0f\n\x0bMODULE_LOAD\x10\x19\x12\x0f\n\x0bMODULE_FREE\x10\x1a\x12\x11\n\rCPU_FREQUENCY\x10\x1b\x62\x06proto3'
)

_PROBETYPE = _descriptor.EnumDescriptor(
  name='ProbeType',
  full_name='ProbeType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_OPENAT', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_READ', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_WRITE', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_UNLINK', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_CHMOD', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_CHOWN', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_EXECVE', index=7, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_FORK', index=8, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_CLONE', index=9, number=9,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_KILL', index=10, number=10,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_EXIT_GROUP', index=11, number=11,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_SOCKET', index=12, number=12,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_CONNECT', index=13, number=13,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_BIND', index=14, number=14,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_LISTEN', index=15, number=15,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_ACCEPT', index=16, number=16,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_GETUID', index=17, number=17,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_SETUID', index=18, number=18,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_SETHOSTNAME', index=19, number=19,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_MOUNT', index=20, number=20,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SYS_ENTER_UMOUNT', index=21, number=21,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SCHED_PROCESS_FORK', index=22, number=22,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SCHED_PROCESS_EXEC', index=23, number=23,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SCHED_PROCESS_EXIT', index=24, number=24,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MODULE_LOAD', index=25, number=25,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MODULE_FREE', index=26, number=26,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='CPU_FREQUENCY', index=27, number=27,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=113,
  serialized_end=721,
)
_sym_db.RegisterEnumDescriptor(_PROBETYPE)

ProbeType = enum_type_wrapper.EnumTypeWrapper(_PROBETYPE)
UNKNOWN = 0
SYS_ENTER_OPENAT = 1
SYS_ENTER_READ = 2
SYS_ENTER_WRITE = 3
SYS_ENTER_UNLINK = 4
SYS_ENTER_CHMOD = 5
SYS_ENTER_CHOWN = 6
SYS_ENTER_EXECVE = 7
SYS_ENTER_FORK = 8
SYS_ENTER_CLONE = 9
SYS_ENTER_KILL = 10
SYS_ENTER_EXIT_GROUP = 11
SYS_ENTER_SOCKET = 12
SYS_ENTER_CONNECT = 13
SYS_ENTER_BIND = 14
SYS_ENTER_LISTEN = 15
SYS_ENTER_ACCEPT = 16
SYS_ENTER_GETUID = 17
SYS_ENTER_SETUID = 18
SYS_ENTER_SETHOSTNAME = 19
SYS_ENTER_MOUNT = 20
SYS_ENTER_UMOUNT = 21
SCHED_PROCESS_FORK = 22
SCHED_PROCESS_EXEC = 23
SCHED_PROCESS_EXIT = 24
MODULE_LOAD = 25
MODULE_FREE = 26
CPU_FREQUENCY = 27



_SYSCALLEVENT = _descriptor.Descriptor(
  name='SyscallEvent',
  full_name='SyscallEvent',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='SyscallEvent.timestamp', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='probe', full_name='SyscallEvent.probe', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pid', full_name='SyscallEvent.pid', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='comm', full_name='SyscallEvent.comm', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=23,
  serialized_end=110,
)

_SYSCALLEVENT.fields_by_name['probe'].enum_type = _PROBETYPE
DESCRIPTOR.message_types_by_name['SyscallEvent'] = _SYSCALLEVENT
DESCRIPTOR.enum_types_by_name['ProbeType'] = _PROBETYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SyscallEvent = _reflection.GeneratedProtocolMessageType('SyscallEvent', (_message.Message,), {
  'DESCRIPTOR' : _SYSCALLEVENT,
  '__module__' : 'syscall_event_pb2'
  # @@protoc_insertion_point(class_scope:SyscallEvent)
  })
_sym_db.RegisterMessage(SyscallEvent)


# @@protoc_insertion_point(module_scope)