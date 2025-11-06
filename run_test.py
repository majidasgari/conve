#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
اسکریپت ساده برای تست سریع مدل و ذخیره نتایج
"""

import subprocess
import sys
import datetime

def main():
    """
    اجرای تست و ذخیره نتایج در فایل
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"test_results_{timestamp}.txt"
    
    print("=" * 70)
    print("شروع تست مدل ConvE")
    print(f"نتایج در فایل '{output_file}' ذخیره خواهد شد")
    print("=" * 70)
    print()
    
    # اجرای اسکریپت تست
    cmd = [
        'python', 'test_model.py',
        '--model', 'conve',
        '--data', 'FB15k-237',
        '--model-path', 'saved_models/FB15k-237_conve_0.2_0.3.model',
        '--input-drop', '0.2',
        '--hidden-drop', '0.3',
        '--feat-drop', '0.2',
        '--use-bias',
        '--cuda'
    ]
    
    try:
        # اجرا و نمایش خروجی real-time
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("نتایج تست مدل ConvE\n")
            f.write("=" * 70 + "\n")
            f.write(f"زمان اجرا: {datetime.datetime.now()}\n")
            f.write("=" * 70 + "\n\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')  # نمایش در terminal
                f.write(line)  # ذخیره در فایل
            
            process.wait()
            
            if process.returncode == 0:
                print("\n" + "=" * 70)
                print(f"✓ تست با موفقیت انجام شد!")
                print(f"✓ نتایج در '{output_file}' ذخیره شد")
                print("=" * 70)
            else:
                print("\n" + "=" * 70)
                print(f"✗ خطا در اجرای تست (کد خروج: {process.returncode})")
                print("=" * 70)
                sys.exit(1)
                
    except FileNotFoundError:
        print("خطا: فایل test_model.py یافت نشد!")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nتست توسط کاربر متوقف شد")
        sys.exit(1)
    except Exception as e:
        print(f"خطای غیرمنتظره: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
