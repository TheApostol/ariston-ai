import random
from typing import Dict, Any

class IoMTAgent:
    """
    Internet of Medical Things (IoMT) Adherence & Monitoring Agent.
    Analyzes simulated device telemetry to forecast patient adherence 
    and early physiological drift.
    """
    
    def forecast_adherence(self, history: str, telemetry: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predict adherence probability for the next 30 days.
        """
        # Default telemetry if none provided (simulated)
        if not telemetry:
            telemetry = {
                "pillbox_opens_7d": random.randint(3, 7),
                "avg_heart_rate": random.randint(65, 85),
                "steps_daily": random.randint(2000, 8000)
            }
            
        base_adherence = 0.85
        
        # Heuristics based on telemetry
        if telemetry.get("pillbox_opens_7d", 7) < 5:
            base_adherence -= 0.30
            
        if "dementia" in history.lower() or "forgetfulness" in history.lower():
            base_adherence -= 0.20
            
        # Stochastic variance
        final_score = min(1.0, max(0.0, base_adherence + random.uniform(-0.1, 0.05)))
        
        risk_level = "HIGH" if final_score < 0.6 else "MODERATE" if final_score < 0.8 else "LOW"
        
        return {
            "adherence_score": round(final_score, 2),
            "risk_level": risk_level,
            "forecast_period": "30-Day Outlook",
            "telemetry_summary": telemetry,
            "recommendations": [
                "Deploy automated SMS reminders" if risk_level != "LOW" else "Continue current monitoring",
                "Schedule caregiver check-in" if risk_level == "HIGH" else None
            ]
        }

iomt_agent = IoMTAgent()
