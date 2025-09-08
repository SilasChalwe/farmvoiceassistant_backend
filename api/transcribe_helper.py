def dummy_transcribe(filename):
    """
    Simulate transcription of audio file.
    Returns the full farming lesson text, in English and Bemba.
    """

    # Full lesson as a single string
    full_lesson = """
1. Plant maize seeds in well-prepared soil, about 2-3 cm deep and 75 cm apart.
2. Water regularly, especially during dry spells, to ensure healthy growth.
3. Apply fertilizer 2-3 weeks after planting to boost growth.
4. Control weeds by hoeing regularly and remove pests as soon as they appear.
5. Harvest maize when the cobs are dry and leaves turn brown, usually 3-4 months after planting.
6. After harvesting, dry the maize cobs under the sun to prevent mold and store in a safe, dry place.
"""

    # Optionally, make "fail" in filename always return fail
    if "fail" in filename.lower():
        return "Transcription Failed"

    return full_lesson.strip()
