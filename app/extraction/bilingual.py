"""
Bilingual Field Association Layer.
Associates corresponding English and native Urdu OCR field entries without machine translation.
"""

from typing import Dict, Any, Optional


def associate_bilingual_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Associate name_en and name_ur, father_name_en and father_name_ur into bilingual objects.
    """
    results = dict(fields)

    # 1. Associate Name
    name_en = fields.get("name_en", {}).get("value", "") or fields.get("name", {}).get("value", "")
    name_ur = fields.get("name_ur", {}).get("value", "")

    if isinstance(name_en, dict):
        name_en_str = name_en.get("en", "")
    else:
        name_en_str = str(name_en)

    if isinstance(name_ur, dict):
        name_ur_str = name_ur.get("ur", "")
    else:
        name_ur_str = str(name_ur)

    if name_en_str or name_ur_str:
        results["name"] = {
            "en": name_en_str,
            "ur": name_ur_str
        }

    # 2. Associate Father Name
    father_en = fields.get("father_name_en", {}).get("value", "") or fields.get("father_name", {}).get("value", "")
    father_ur = fields.get("father_name_ur", {}).get("value", "")

    if isinstance(father_en, dict):
        father_en_str = father_en.get("en", "")
    else:
        father_en_str = str(father_en)

    if isinstance(father_ur, dict):
        father_ur_str = father_ur.get("ur", "")
    else:
        father_ur_str = str(father_ur)

    if father_en_str or father_ur_str:
        results["father_name"] = {
            "en": father_en_str,
            "ur": father_ur_str
        }

    return results
