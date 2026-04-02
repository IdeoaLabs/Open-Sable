# Calendar Module - Timezone Handling Guide

## Overview
Timezone support is critical for global calendar systems. This guide covers advanced timezone handling patterns.

## Key Concepts
``python
from datetime import datetime, tzinfo, timedelta
import pytz
class UTC(tzinfo):
    """
zona horaria universal (UTC+)
    """
def utcoffset(self, dt):
    return timedelta(0)
def dst(self, dt):
    # No daylight saving time in UTC
    return timedelta(0)def tzname(self, dt):
    return "Z"
```
