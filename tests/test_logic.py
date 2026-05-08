import sys
sys.path.insert(0, '.')
from app.app_enhanced import predict_stress

# Test with ALL NORMAL VALUES + Calm psych
test_data = {
    'Heart_Rate': '75',
    'HRV': '60',
    'Respiration': '16',
    'Skin_Temp': '36.8',
    'BP_Systolic': '115',
    'BP_Diastolic': '75',
    'Cognitive_State': '2',
    'Emotional_State': '2'
}

result = predict_stress(test_data)
print('TEST 1: All Normal + Cognitive=2, Emotional=2')
print('  Condition:', result.get('condition'))
print('  Stress Level:', result.get('stress_level'))
print('  MLI:', result.get('mental_load_index'))
print()

# Test with moderate values
test_data2 = {
    'Heart_Rate': '105',
    'HRV': '40',
    'Respiration': '22',
    'Skin_Temp': '37.0',
    'BP_Systolic': '130',
    'BP_Diastolic': '85',
    'Cognitive_State': '3',
    'Emotional_State': '3'
}

result2 = predict_stress(test_data2)
print('TEST 2: Moderate values + Cognitive=3, Emotional=3')
print('  Condition:', result2.get('condition'))
print('  Stress Level:', result2.get('stress_level'))
print('  MLI:', result2.get('mental_load_index'))
