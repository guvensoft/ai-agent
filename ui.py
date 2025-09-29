# ui.py (güncellenmiş — unidiff ile hunk ayrıştırma ve esnek apply)
import streamlit as st
import requests, json
from pathlib import Path

st.set_page_config(page_title="AI Dev Agent - Hunk UI", layout="wide")
st.title("AI Dev Agent — Patch Preview & Hunk Selection")

API_URL = st.text_input("API URL", "http://localhost:8000").rstrip("/")

if "last_plan" not in st.session_state:
    st.session_state.last_plan = {}
if "hunks" not in st.session_state:
    st.session_state.hunks = []
if "selections" not in st.session_state:
    st.session_state.selections = {}

st.header("1) Plan Oluştur")
plan_req = st.text_area("Plan isteği", "API'ye /healthz ekle, basit JSON dönsün ve unit test yaz.")
if st.button("Plan Oluştur"):
    try:
        r = requests.post(f"{API_URL}/dev/plan", json={"request": plan_req}, timeout=180)
        st.write("HTTP", r.status_code)
        if r.ok:
            st.session_state.last_plan = r.json()
            st.success("Plan oluşturuldu: " + st.session_state.last_plan.get("plan_id","?"))
            # hemen patch'i parçala
            patch_text = st.session_state.last_plan.get("patch","")
            st.session_state.hunks = []  # reset
            if patch_text:
                st.session_state.hunks = []
                # try unidiff
                try:
                    from unidiff import PatchSet
                    ps = PatchSet(patch_text.splitlines(True))
                    for pf in ps:
                        file_path = pf.path or pf.target_file or pf.source_file
                        file_hunks = []
                        for h in pf:
                            # h may not cast to string nicely; use join of h._lines
                            try:
                                htext = "".join(h._lines)
                            except Exception:
                                # fallback: reconstruct from hunks header + lines
                                h_lines = []
                                for ln in h:
                                    # ln may be UnidiffLine object with value attr
                                    try:
                                        h_lines.append(ln.value)
                                    except Exception:
                                        h_lines.append(str(ln))
                                htext = "\n".join(h_lines)
                            file_hunks.append(htext)
                        st.session_state.hunks.append({"file": file_path, "hunks": file_hunks})
                except Exception:
                    # fallback: simple split by 'diff --git' and then '@@' markers
                    parts = patch_text.split("\ndiff --git ")
                    for p in parts:
                        if not p.strip():
                            continue
                        if not p.startswith("diff --git "):
                            p = "diff --git " + p
                        # get filename from header
                        first_line = p.splitlines()[0]
                        # attempt to find a/b/filename
                        import re
                        m = re.search(r"a/(.*?) b/(.*?)\\b", first_line)
                        # safer: find ' b/' token
                        fname = None
                        lines = p.splitlines()
                        header = lines[0]
                        # heuristics:
                        if " b/" in header:
                            try:
                                fname = header.split(" b/")[-1].split()[0]
                            except Exception:
                                fname = header
                        else:
                            fname = header
                        hunks = []
                        cur = []
                        for ln in lines:
                            if ln.startswith("@@"):
                                if cur:
                                    hunks.append("\n".join(cur))
                                cur = [ln]
                            else:
                                cur.append(ln)
                        if cur:
                            hunks.append("\n".join(cur))
                        st.session_state.hunks.append({"file": fname, "hunks": hunks})

        else:
            st.error(f"Plan oluşturma hatası: {r.status_code} -- {r.text[:2000]}")
    except Exception as e:
        st.exception(e)

if st.session_state.last_plan:
    plan_id = st.session_state.last_plan.get("plan_id")
    st.subheader("Plan: " + str(plan_id))
    st.code(st.session_state.last_plan.get("plan",""), language="markdown")
    st.subheader("Patch (tam)")
    st.code(st.session_state.last_plan.get("patch","")[:20000], language="diff")

    st.markdown("----")
    # Sandbox test (tam patch)
    if st.button("Sandbox Test (tam patch)"):
        if not plan_id:
            st.warning("Önce plan oluştur.")
        else:
            try:
                r = requests.post(f"{API_URL}/dev/sandbox_test", json={"plan_id": plan_id}, timeout=300)
                st.write("HTTP", r.status_code)
                try:
                    st.json(r.json())
                except Exception:
                    st.text(r.text)
            except Exception as e:
                st.exception(e)

    # If hunks parsed, show them and selection boxes
    if st.session_state.hunks:
        st.subheader("Dosya ve Hunk Seçimi")
        selections = {}
        for file_block in st.session_state.hunks:
            file = file_block.get("file")
            st.markdown(f"**{file}**")
            hlist = file_block.get("hunks", [])
            sel = []
            for i, h in enumerate(hlist):
                key = f"{plan_id}::{file}::hunk::{i}"
                checked = st.checkbox(f"Hunk {i} (file: {file})", key=key, value=False)
                # show a small preview of hunk
                st.code(h[:2000], language="diff")
                if checked:
                    sel.append(i)
            if sel:
                selections[file] = sel
        st.session_state.selections = selections

        st.markdown("---")
        st.caption("Seçili hunks/filenames map'i sunucuya gönderilecek. Eğer sunucu sadece file-seçimi destekliyorsa, hunk bilgisi göz ardı edilebilir.")

    # Apply selected hunks (try hunks endpoint first, fallback to files endpoint)
    if st.button("Seçili Hunk'leri Uygula ve Test Çalıştır"):
        selections = st.session_state.get("selections", {})
        if not selections:
            st.warning("Hiç hunk seçilmedi.")
        else:
            # first try hunk-level endpoint (if server supports)
            try:
                r = requests.post(f"{API_URL}/dev/apply/hunks", json={"plan_id": plan_id, "selections": selections}, timeout=180)
                st.write("Tried /dev/apply/hunks -> HTTP", r.status_code)
                if r.ok:
                    st.success("Uygulandı (hunk-level).")
                    try:
                        res = r.json()
                        st.json(res)
                    except Exception:
                        st.text(r.text)
                else:
                    # if not allowed or not found, fallback to files-level apply
                    st.warning(f"/dev/apply/hunks returned {r.status_code}: {r.text[:1000]}")
                    # fallback: send files list only
                    files_only = list(selections.keys())
                    r2 = requests.post(f"{API_URL}/dev/apply/files", json={"plan_id": plan_id, "files": files_only}, timeout=180)
                    st.write("Fallback /dev/apply/files -> HTTP", r2.status_code)
                    if r2.ok:
                        st.success("Uygulandı (file-level).")
                        try:
                            res2 = r2.json()
                            st.json(res2)
                        except Exception:
                            st.text(r2.text)
                    else:
                        st.error(f"Fallback başarısız: {r2.status_code}: {r2.text[:2000]}")
            except requests.exceptions.RequestException as e:
                st.exception(e)

    st.markdown("---")
    st.caption("Not: apply endpoint'leri çalışma dizininizin temiz olmasını gerektirir. Hataları görmek için sunucu loglarına bakın veya API cevaplarını inceleyin.")
