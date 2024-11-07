#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
from datetime import datetime

log_filename = fr"log_files/log_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    filename = log_filename,
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'    
)

# 공통 로거 인스턴스 생성
logger = logging.getLogger('common_logger')

