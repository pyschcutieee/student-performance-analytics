
def generate_recommendations(student, risk_level):
    recommendations = []

    if student["studytime"] <= 2:
        recommendations.append(
            "Increase focused study time gradually and follow a daily routine."
        )

    if student["failures"] > 0:
        recommendations.append(
            "Revise difficult concepts and practice previous exam questions."
        )

    if student["absentees"] > 10:
        recommendations.append(
            "Improve attendance and complete missed classwork."
        )

    if student["G1"] < 10 or student["G2"] < 10:
        recommendations.append(
            "Prioritize weak topics and take weekly practice tests."
        )

    if not recommendations:
        recommendations.append(
            "Maintain consistent study habits and revise regularly."
        )

    return recommendations


def generate_study_plan(student, risk_level):
    risk = str(risk_level).lower()

    if "high" in risk:
        return [
            ("4:00 PM - 5:00 PM", "Weak subject revision"),
            ("5:00 PM - 5:30 PM", "Break and physical activity"),
            ("5:30 PM - 6:30 PM", "Homework and practice"),
            ("7:00 PM - 8:00 PM", "Mock tests and revision")
        ]

    elif "medium" in risk:
        return [
            ("4:00 PM - 5:00 PM", "Concept revision"),
            ("5:00 PM - 5:30 PM", "Break"),
            ("5:30 PM - 6:30 PM", "Assignments and practice"),
            ("7:00 PM - 7:30 PM", "Quick revision")
        ]

    else:
        return [
            ("4:00 PM - 5:00 PM", "Homework and assignments"),
            ("5:00 PM - 5:30 PM", "Break"),
            ("5:30 PM - 6:30 PM", "Revision and advanced practice")
        ]