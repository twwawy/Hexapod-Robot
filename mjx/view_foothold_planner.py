#!/usr/bin/env python3
"""Stable launcher for the keyboard foothold explorer."""
import sys


def main():
    # Route before importing the legacy viewer and its archived policy modules.
    # The adaptive replay uses the actual MJX training terrain and 24-D contract.
    adaptive = '--controller=adaptive' in sys.argv
    if adaptive:
        sys.argv.remove('--controller=adaptive')
    elif '--controller' in sys.argv:
        index = sys.argv.index('--controller')
        if index+1 < len(sys.argv) and sys.argv[index+1] == 'adaptive':
            del sys.argv[index:index+2]
            adaptive = True
    if adaptive:
        from view_adaptive_gait import main as run
    else:
        from view_foothold_explorer import main as run
    run()

if __name__ == "__main__":
    main()
