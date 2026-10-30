# recommendations.py
import json
from pathlib import Path

DISCLAIMER = (
    "This is an AI estimate from a photo of the visible surface only. It cannot "
    "detect internal faults. Always follow your panel manufacturer's instructions "
    "and consult a certified solar technician when unsure."
)

# Keys MUST match models/class_names.json exactly
RECOMMENDATIONS = {
    "Clean": {
        "description": "No visible dust, droppings, snow or damage was detected on the panel surface. Note: this is a visual check only; internal faults are not detectable from a normal photo.",
        "severity": "Low",
        "urgency": "None (routine maintenance)",
        "actions": [
            "No action required.",
            "Keep a routine inspection schedule (for example every 3 to 6 months).",
        ],
    },
    "Dusty": {
        "description": "A layer of dust, sand or grime on the glass is blocking sunlight and reducing energy output.",
        "severity": "Low",
        "urgency": "Within 1 to 2 weeks",
        "actions": [
            "Check the panel manufacturer's cleaning instructions first, and shut the system down before cleaning if they say so.",
            "Clean from the ground with a soft brush or sponge on a pole and clean water. Avoid climbing onto the roof.",
            "Clean in the early morning or late evening when panels are cool. Never pour cold water on hot panels, and avoid high-pressure jets.",
            "Avoid abrasive tools and harsh chemicals; they scratch the glass and cause shadows.",
        ],
    },
    "Bird-drop": {
        "description": "Bird droppings create opaque spots that shade parts of the panel. Persistent shading can cause uneven heating (hot spots) and lower output.",
        "severity": "Medium",
        "urgency": "Within 3 to 5 days",
        "actions": [
            "Soak the area with clean water to soften dried droppings, then wipe gently with a soft, non-abrasive sponge.",
            "Use plain water first. Use mild soap only if the manufacturer allows it, and rinse off all residue.",
            "Clean from the ground when possible and do not climb onto the roof. Hire a professional cleaner if the panels are hard to reach.",
            "If the same spot gets soiled repeatedly, consider bird-deterrent measures installed by a professional.",
        ],
    },
    "Snow-Covered": {
        "description": "Snow on the panel blocks sunlight and stops power generation until it clears.",
        "severity": "Low to Medium",
        "urgency": "No rush; snow often slides off or melts by itself",
        "actions": [
            "Wait for natural melting or sliding when possible, especially if the covering is short-lived.",
            "If you must clear it, use a soft long-handled brush or foam rake from the ground. Never climb onto a snowy or icy roof.",
            "Never use metal tools, sharp objects, or hot water; temperature shock can crack the glass.",
            "Check your panel's rated snow load in its datasheet, especially for heavy snowfall.",
        ],
    },
    "Electrical-damage": {
        "description": "Visible signs of electrical trouble were detected, such as burn marks, discoloration, or melted connectors or junction box. This can be a safety and fire hazard.",
        "severity": "High",
        "urgency": "Immediate: contact a professional",
        "actions": [
            "Stay away from the panel and wiring. Do not touch it or try a DIY repair. Panels can produce dangerous DC voltage whenever there is light, even if the system is switched off.",
            "Switch the system off at the inverter only if the manufacturer's instructions allow it and you can do it without going near the damaged part.",
            "If you see smoke, sparks or smell burning, move away and call emergency services (112 in India).",
            "Contact a certified solar technician for safety testing and repair or replacement.",
        ],
    },
    "Physical-Damage": {
        "description": "Visible structural damage was detected, such as cracked or broken glass, a deformed frame, or a torn backsheet, usually from impact, hail or stress.",
        "severity": "High",
        "urgency": "Immediate: contact a professional",
        "actions": [
            "Do not touch the damaged surface. Broken glass can cut, and damaged panels can still carry voltage, especially if wet.",
            "Keep people and pets away from the area, and switch the system off at the inverter only if it is safe and the manufacturer allows it.",
            "Moisture entering cracks can cause short circuits, so avoid spraying water on the panel.",
            "Call a certified solar technician or your installer to inspect and replace the module.",
        ],
    },
    "Not-a-panel": {
    "description": "The uploaded image does not appear to be a photovoltaic solar panel. It may be a room, outdoor scene, face, or general object.",
    "severity": "N/A",
    "urgency": "None",
    "recommendation": "Please upload a valid solar panel image for defect analysis."
}
}


def get_recommendation(class_name):
    """Never raises KeyError; falls back to a safe default."""
    return RECOMMENDATIONS.get(class_name, {
        "description": "No specific guidance is available for this result.",
        "severity": "Unknown",
        "urgency": "Consult a certified solar technician",
        "actions": ["Contact a certified solar technician for an inspection."],
    })


def check_keys(class_names_path="models/class_names.json"):
    class_names = json.loads(Path(class_names_path).read_text())
    missing = [c for c in class_names if c not in RECOMMENDATIONS]
    extra = [k for k in RECOMMENDATIONS if k not in class_names]
    assert not missing and not extra, f"missing={missing}, extra={extra}"
    print("OK: all", len(class_names), "classes have recommendations")


if __name__ == "__main__":
    check_keys()