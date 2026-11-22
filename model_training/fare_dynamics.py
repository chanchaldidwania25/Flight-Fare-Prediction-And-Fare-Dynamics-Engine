def fare_dynamics_engine(
    current_fare,
    predicted_fare
):

    difference = predicted_fare - current_fare

    percentage_change = (
        difference / current_fare
    ) * 100

    # Recommendation

    if percentage_change > 10:
        recommendation = "BOOK NOW"

    elif percentage_change < -10:
        recommendation = "WAIT"

    else:
        recommendation = "STABLE"

    # Improved Risk Logic

    absolute_change = abs(percentage_change)

    if absolute_change < 5:
        risk_score = 25
        risk_label = "🟢 Low Risk"

    elif absolute_change < 15:
        risk_score = 50
        risk_label = "🟡 Medium Risk"

    elif absolute_change < 25:
        risk_score = 75
        risk_label = "🟠 High Risk"

    else:
        risk_score = 100
        risk_label = "🔴 Very High Risk"

    savings = abs(difference)

    return (
        recommendation,
        percentage_change,
        risk_score,
        risk_label,
        savings
    )