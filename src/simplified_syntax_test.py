"""
Simplified Syntax Test - Verify all frameworks compile correctly
"""

import sys
sys.path.append('src')

def test_framework_syntax():
    """Test syntax validation for all frameworks."""
    
    print('🔍 AUTONOMOUS RESEARCH SYSTEM - SYNTAX VALIDATION')
    print('=' * 60)
    
    frameworks = [
        ('Agentic Sentiment Framework', 'autonomous_agentic_sentiment_framework'),
        ('Research Validation System', 'autonomous_research_validation_system'), 
        ('Research Benchmarking Suite', 'autonomous_research_benchmarking_suite'),
        ('Master Research System', 'autonomous_research_master_system')
    ]
    
    passed_count = 0
    
    for name, module_name in frameworks:
        try:
            # Try to compile the file
            with open(f'{module_name}.py', 'r') as f:
                code = f.read()
            
            # Attempt to compile
            compile(code, f'{module_name}.py', 'exec')
            print(f'✅ {name} - Syntax Validation PASSED')
            passed_count += 1
            
        except SyntaxError as e:
            print(f'❌ {name} - Syntax Error: Line {e.lineno}, {e.msg}')
        except Exception as e:
            print(f'⚠️  {name} - Other Error: {str(e)[:100]}')
    
    print(f'\n🎯 Quality Gate: Syntax Validation - {passed_count}/{len(frameworks)} frameworks passed')
    
    if passed_count == len(frameworks):
        print('✅ All frameworks passed syntax validation!')
        return True
    else:
        print('❌ Some frameworks failed syntax validation.')
        return False

if __name__ == "__main__":
    test_framework_syntax()