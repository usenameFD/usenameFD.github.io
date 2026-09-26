"""Date Governance (CORRECTION #6)"""
import pandas as pd
from typing import List, Optional

class ComputationDateGovernance:
    """Pre-announced, auditable DRC computation dates."""
    
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 frequency: str = 'W-FRI',
                 holiday_calendar: Optional[List[str]] = None):
        """
        CORRECTION #6: Pre-announced governance calendar
        
        Regulatory requirement (CRR3): "day of computation should be carefully selected"
        = Pre-announced, not post-hoc, not data-dependent
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.frequency = frequency
        self.holidays = set(pd.to_datetime(holiday_calendar)) if holiday_calendar else set()
        self._generate_calendar()
    
    def _generate_calendar(self) -> None:
        """Generate pre-defined computation dates (NOT data-dependent)."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq=self.frequency)
        self.computation_dates = [d for d in dates if d not in self.holidays]
    
    def validate_date(self, date: pd.Timestamp) -> bool:
        """Check if date is in pre-announced calendar."""
        if date not in self.computation_dates:
            raise ValueError(f"Date {date} not in governance calendar!")
        return True
