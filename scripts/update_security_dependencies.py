#!/usr/bin/env python3
"""
Security Dependency Update Script

This script identifies and updates vulnerable dependencies to secure versions.
Handles system-installed packages by providing user-space alternatives.
"""

import subprocess
import sys
from pathlib import Path
from packaging import version
import json


def get_current_version(package_name):
    """Get currently installed version of a package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return None


def install_user_package(package_spec):
    """Install package in user space to avoid system conflicts."""
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--user", "--upgrade", package_spec]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def main():
    """Update security dependencies to fix known vulnerabilities."""
    
    # Security requirements mapping
    security_updates = {
        "setuptools": {
            "min_version": "78.1.1",
            "cves": ["CVE-2025-47273", "CVE-2024-6345"],
            "spec": "setuptools>=78.1.1"
        },
        "cryptography": {
            "min_version": "42.0.5", 
            "cves": ["Multiple CVEs"],
            "spec": "cryptography>=42.0.5"
        },
        "pip": {
            "min_version": "25.0",
            "cves": ["PVE-2025-75180"],
            "spec": "pip>=25.0"
        },
        "PyJWT": {
            "min_version": "2.10.1",
            "cves": ["CVE-2024-53861"],
            "spec": "PyJWT>=2.10.1"
        }
    }
    
    print("🔒 Security Dependency Update Script")
    print("=" * 40)
    
    updates_needed = []
    updates_successful = []
    updates_failed = []
    
    # Check current versions and update needs
    for package, info in security_updates.items():
        current_version = get_current_version(package)
        if current_version:
            if version.parse(current_version) < version.parse(info["min_version"]):
                updates_needed.append((package, current_version, info))
                print(f"⚠️  {package} {current_version} < {info['min_version']} (vulnerable to {', '.join(info['cves'])})")
            else:
                print(f"✅ {package} {current_version} >= {info['min_version']} (secure)")
        else:
            print(f"❓ {package} not found or not directly importable")
    
    if not updates_needed:
        print("\n🎉 All dependencies are up to date!")
        return 0
    
    print(f"\n📦 Attempting to update {len(updates_needed)} vulnerable packages...")
    
    # Attempt updates
    for package, current_ver, info in updates_needed:
        print(f"\n🔄 Updating {package} from {current_ver} to >={info['min_version']}...")
        
        success, stdout, stderr = install_user_package(info["spec"])
        
        if success:
            new_version = get_current_version(package)
            if new_version and version.parse(new_version) >= version.parse(info["min_version"]):
                updates_successful.append((package, current_ver, new_version))
                print(f"✅ Successfully updated {package}: {current_ver} → {new_version}")
            else:
                updates_failed.append((package, current_ver, "Version check failed"))
                print(f"❌ Update may have failed - version check inconclusive")
        else:
            updates_failed.append((package, current_ver, stderr))
            print(f"❌ Failed to update {package}: {stderr}")
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 UPDATE SUMMARY")
    print("=" * 40)
    
    if updates_successful:
        print(f"✅ Successfully updated ({len(updates_successful)}):")
        for package, old_ver, new_ver in updates_successful:
            print(f"   • {package}: {old_ver} → {new_ver}")
    
    if updates_failed:
        print(f"\n❌ Failed updates ({len(updates_failed)}):")
        for package, old_ver, error in updates_failed:
            print(f"   • {package} ({old_ver}): {error}")
        
        print("\n🔧 MANUAL UPDATE INSTRUCTIONS:")
        print("For system-installed packages that failed, try:")
        print("1. Create a virtual environment: python -m venv .venv")
        print("2. Activate it: source .venv/bin/activate")
        print("3. Install updated packages: pip install setuptools>=78.1.1 cryptography>=42.0.5")
        print("4. Or use pip install --user <package_spec> to install in user space")
    
    # Update requirements.txt if needed
    if updates_successful:
        print(f"\n📝 Consider updating pyproject.toml with minimum secure versions")
    
    return len(updates_failed)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)