#!/usr/bin/env python3
"""
SSH connection script that monitors DUO_OTP environment variable for changes
and handles 2FA authentication when connecting to BALAM cluster.
"""

import os
import subprocess
import time
import pexpect


def get_current_otp():
    """Get the current DUO_OTP value from environment."""
    return os.getenv("DUO_OTP", "")


def wait_for_otp_change(initial_otp, timeout=300):
    """Wait for DUO_OTP to change from initial value."""
    print(f"Waiting for DUO_OTP to change from current value: {initial_otp}")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        current_otp = get_current_otp()
        if current_otp != initial_otp and current_otp:
            print(f"OTP changed to: {current_otp}")
            return current_otp
        time.sleep(2)  # Check every 2 seconds
    
    raise TimeoutError(f"OTP did not change within {timeout} seconds")


def ssh_with_duo_auth(hostname="balam.scinet.utoronto.ca", username="sgbaird"):
    """
    Connect to SSH with Duo 2FA authentication.
    Monitors DUO_OTP environment variable for new codes.
    """
    initial_otp = get_current_otp()
    print(f"Starting SSH connection to {username}@{hostname}")
    print(f"Initial DUO_OTP value: {initial_otp}")
    
    try:
        # Start SSH connection with pexpect for interaction
        ssh_cmd = f"ssh -o StrictHostKeyChecking=no {username}@{hostname}"
        child = pexpect.spawn(ssh_cmd, timeout=60)
        
        # Enable logging to see what's happening
        child.logfile_read = open('/tmp/ssh_debug.log', 'wb')
        
        # Wait for different possible prompts
        index = child.expect([
            pexpect.TIMEOUT,
            "Permission denied",
            "Duo two-factor login",
            "Enter a passcode",
            "Passcode:",
            r"\$ ",  # Shell prompt if successful
            pexpect.EOF
        ])
        
        if index == 0:  # TIMEOUT
            print("SSH connection timed out")
            return False
        elif index == 1:  # Permission denied
            print("Permission denied - SSH key authentication failed")
            return False
        elif index in [2, 3, 4]:  # Duo prompts
            print("Duo 2FA prompt detected, waiting for new OTP...")
            try:
                new_otp = wait_for_otp_change(initial_otp)
                print(f"Sending OTP: {new_otp}")
                child.sendline(new_otp)
                
                # Wait for login success
                index = child.expect([
                    pexpect.TIMEOUT,
                    "Permission denied",
                    "Invalid passcode",
                    r"\$ ",  # Shell prompt
                    r"[#$] "  # Alternative shell prompt
                ], timeout=30)
                
                if index in [3, 4]:  # Success
                    print("SSH connection successful!")
                    return child
                else:
                    print(f"Login failed after OTP submission (index: {index})")
                    return False
                    
            except TimeoutError as e:
                print(f"Error: {e}")
                return False
        elif index == 5:  # Direct shell access (no 2FA needed)
            print("SSH connection successful without 2FA!")
            return child
        else:  # EOF
            print("SSH connection ended unexpectedly")
            return False
            
    except Exception as e:
        print(f"SSH connection error: {e}")
        return False
    finally:
        if 'child' in locals():
            try:
                child.logfile_read.close()
            except:
                pass


def test_balam_connection():
    """Test the BALAM connection and run basic commands."""
    connection = ssh_with_duo_auth()
    
    if connection:
        try:
            # Run some basic commands to verify the connection
            print("\n=== Testing BALAM connection ===")
            
            # Check hostname
            connection.sendline("hostname")
            connection.expect(r"[#$] ")
            print("Hostname command executed")
            
            # Check SLURM
            connection.sendline("sinfo --version")
            connection.expect(r"[#$] ")
            print("SLURM version check executed")
            
            # Check modules
            connection.sendline("module avail 2>&1 | head -5")
            connection.expect(r"[#$] ")
            print("Module availability check executed")
            
            # Close connection
            connection.sendline("exit")
            connection.close()
            print("Connection closed successfully")
            return True
            
        except Exception as e:
            print(f"Error during connection test: {e}")
            try:
                connection.close()
            except:
                pass
            return False
    else:
        print("Failed to establish SSH connection")
        return False


if __name__ == "__main__":
    print("BALAM SSH Connection Test with Duo 2FA")
    print("=" * 50)
    
    # Check for required dependencies
    try:
        import pexpect
    except ImportError:
        print("Error: pexpect module not installed. Install with: pip install pexpect")
        exit(1)
    
    success = test_balam_connection()
    exit(0 if success else 1)