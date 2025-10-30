#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索系统启动脚本
提供多种启动选项
"""

import sys
import os

def show_menu():
    """显示启动菜单"""
    print("🎯 智能中医问答系统 (v4.0)")
    print("=" * 30)
    print("架构：智能路由 + 向量检索 + 知识图谱")
    print("=" * 30)
    print("请选择启动模式:")
    print("1. 完整版 (推荐) - 包含所有功能")
    print("2. 简化版 - 快速启动")
    print("3. 帮助信息")
    print("4. 退出")
    print()

def show_help():
    """显示帮助信息"""
    print("\n📖 系统帮助")
    print("=" * 30)
    print("🎯 系统功能 (v4.0):")
    print("   • BERT智能路由分类")
    print("   • 向量语义检索")
    print("   • 知识图谱检索")
    print("   • 自适应混合检索")
    print()
    print("💡 查询示例:")
    print("   • 头痛治疗")
    print("   • 人参功效")
    print("   • 感冒发烧")
    print("   • 失眠调理")
    print("   • 四君子汤组成")
    print()
    print("🔧 启动模式:")
    print("   • 完整版: 包含统计、健康检查等功能")
    print("   • 简化版: 仅核心查询功能，启动更快")
    print()

def main():
    """主函数"""
    while True:
        show_menu()
        
        try:
            choice = input("请输入选择 (1-4): ").strip()
            
            if choice == '1':
                print("\n🚀 启动完整版系统...")
                os.system("python main.py")
                break
            elif choice == '2':
                print("\n🚀 启动简化版系统...")
                os.system("python simple_main.py")
                break
            elif choice == '3':
                show_help()
                input("\n按回车键继续...")
            elif choice == '4':
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入 1-4")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {str(e)}")

if __name__ == "__main__":
    main()
