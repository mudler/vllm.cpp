ID: ISSUE-LOCAL-01M2A7P3C95W3PBAVT9SC6KKY5
Title: doctest renders a const char* operand as 1, so every CAPTURE/MESSAGE device tag in test_backend_cross_device.cpp is unreadable
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

MESSAGE(a << b) expands to mb * a << b and MessageBuilder stringifies through doctest::toString; for a const char* const& operand that resolves to the bool overload and prints 1. CAPTURE takes the same path. tests/vt/test_backend_cross_device.cpp passes DeviceName(dt) raw at roughly forty CAPTURE sites, so a failing cross-device assertion logs 'DeviceName(dt) := 1' and cannot say which device failed. The three MODEL-MM-QWEN4-EXP W1 cases were repaired in place with a DeviceTag() helper that returns std::string; the remaining sites, which belong to other rows, are not touched.

## Resolution

-
