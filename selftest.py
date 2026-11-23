"""Self-check: parser asserts + one real generation using both models. Run: .venv\\Scripts\\python selftest.py"""
import server as s

cfg = s.tags()
segs, warns = s.parse("Hi. [excited] Wow! [laugh] [adam] [whisper] Shh. [bogus] ok", cfg, "heart")
assert segs == [("heart", "neutral", "Hi."), ("heart", "excited", "Wow! [laugh]"), ("adam", "whisper", "Shh.  ok")], segs
assert warns == ["Unknown tag [bogus] ignored"], warns
assert s.chunks("A. " * 200, 250) and all(len(c) <= 250 for c in s.chunks("Short one. " * 100))
print("parser OK")

s.jobs["t"] = {"done": False, "progress": 0, "status": "", "warnings": []}
s.run("t", s.Req(text="[calm] Good evening. [excited] We actually won the whole thing! [laugh] "
                      "[george] [whisper] Don't tell anyone.", voice="bella"))
j = s.jobs["t"]
assert not j.get("error"), j
print("generation OK:", j["wav"], j["mp3"], j["seconds"], "s")
