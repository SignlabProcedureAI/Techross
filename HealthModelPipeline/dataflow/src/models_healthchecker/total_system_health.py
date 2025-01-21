# basic
import pandas as pd
import time

# module.healthchecker
from .csu_system_health import ModelCsuSystemHealth
from .sts_system_health import ModelStsSystemHealth
from .fts_system_health import ModelFtsSystemHealth
from .fmu_system_health import ModelFmuSystemHealth
from .tro_fault_detector import TROFaultAlgorithm
from .current_system_health import ModelCurrentystemHealth

def time_decorator(func): 
    """
    함수의 실행 시간을 측정하는 데코레이터.

    Args:
        func (callable): 데코레이터가 적용될 함수.

    Returns:
        callable: 실행 시간을 측정하고 결과를 반환하는 래퍼 함수
    """
    def wrapper(*args, **kwargs):
        start_time  = time.time() # 시작 시간 기록
        result = func(*args, **kwargs) # 함수 실행
        end_time = time.time() # 종료 시간 기록
        print(f"{func.__name__} 함수 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper

@time_decorator
def apply_system_health_learning_algorithms_with_total(data: pd.DataFrame, ship_id: str) -> None:
    """
    시스템 건강 알고리즘을 순차적으로 적용하는 함수.

    Args:
        data (pd.DataFrame): 입력 데이터 프레임.
        ship_id (str): 선박 ID.
        op_index (int): 작업 인덱스.
        section (str): 섹션 정보.
    """
    # 클래스와 메서드 정의를 리스트로 관리
    algorithms = [
        (ModelCsuSystemHealth, "apply_system_health_algorithms_with_csu"),
        (ModelStsSystemHealth, "apply_system_health_algorithms_with_sts"),
        (TROFaultAlgorithm, "apply_tro_fault_detector"),
        (ModelFtsSystemHealth, "apply_system_health_algorithms_with_fts"),
        (ModelFmuSystemHealth, "apply_system_health_algorithms_with_fmu"),
        (ModelCurrentystemHealth, "apply_system_health_algorithms_with_current")
    ]

    for model_class, method_name in algorithms:
        if method_name == 'apply_system_health_algorithms_with_fmu':
            instance = model_class(data, ship_id=ship_id)
        else:
            instance = model_class(data)
        getattr(instance, method_name)(status=True)