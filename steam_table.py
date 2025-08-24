# ...existing code...
"""
steam_table.py
-------------------
A script for identifying steam (water) properties using CoolProp and Streamlit.
Includes helpers for unit conversion, property calculation, input validation, and plotting.
"""

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI, PhaseSI

# ---------- Unit helpers ----------
def to_SI_T(t_val, t_unit):
    return t_val + 273.15 if t_unit == "°C" else t_val

def from_SI_T(TK, t_unit):
    return TK - 273.15 if t_unit == "°C" else TK

def to_SI_P(p_val, p_unit):
    factor = {"Pa":1.0, "kPa":1e3, "bar":1e5, "MPa":1e6}[p_unit]
    return p_val * factor

def from_SI_P(Pa, p_unit):
    factor = {"Pa":1.0, "kPa":1e-3, "bar":1e-5, "MPa":1e-6}[p_unit]
    return Pa * factor

def is_water(fluid):
    return fluid.lower() == "water"

# ---------- Saturation helpers (Water) ----------
def water_crit():
    try:
        Tc = PropsSI("Tcrit", "Water")
        Pc = PropsSI("pcrit", "Water")
        rhoc = PropsSI("rhocrit", "Water")
        vc = 1.0 / rhoc
        return Tc, Pc, vc
    except Exception:
        return np.nan, np.nan, np.nan

def water_triple():
    try:
        Tt = PropsSI("Ttriple", "Water")
        Pt = PropsSI("ptriple", "Water")
        return Tt, Pt
    except Exception:
        return np.nan, np.nan

def sat_curves_Ts_water(n=180):
    try:
        Tt, _ = water_triple()
        Tc, _, _ = water_crit()
        Tgrid = np.linspace(Tt, Tc * 0.9995, n)
        sL, sV = [], []
        for T in Tgrid:
            try:
                sL.append(PropsSI("S", "T", T, "Q", 0, "Water") / 1000.0)
                sV.append(PropsSI("S", "T", T, "Q", 1, "Water") / 1000.0)
            except Exception:
                sL.append(np.nan); sV.append(np.nan)
        return Tgrid, np.array(sL), np.array(sV)
    except Exception:
        return np.array([]), np.array([]), np.array([])

def sat_curves_Ph_water(n=200):
    try:
        Tt, _ = water_triple()
        Tc, _, _ = water_crit()
        Tgrid = np.geomspace(max(Tt, 1e-6), Tc * 0.9995, n)
        P, hL, hV = [], [], []
        for T in Tgrid:
            try:
                P_sat = PropsSI("P", "T", T, "Q", 0, "Water")
                P.append(P_sat)
                hL.append(PropsSI("H", "T", T, "Q", 0, "Water") / 1000.0)
                hV.append(PropsSI("H", "T", T, "Q", 1, "Water") / 1000.0)
            except Exception:
                P.append(np.nan); hL.append(np.nan); hV.append(np.nan)
        return np.array(P), np.array(hL), np.array(hV)
    except Exception:
        return np.array([]), np.array([]), np.array([])

def sat_curves_Pv_water(n=220):
    try:
        Tt, _ = water_triple()
        Tc, _, _ = water_crit()
        Tgrid = np.geomspace(max(Tt, 1e-6), Tc * 0.9995, n)
        P, vL, vV = [], [], []
        for T in Tgrid:
            try:
                P_sat = PropsSI("P", "T", T, "Q", 0, "Water")
                rhoL = PropsSI("D", "T", T, "Q", 0, "Water")
                rhoV = PropsSI("D", "T", T, "Q", 1, "Water")
                P.append(P_sat)
                vL.append(1.0 / rhoL)
                vV.append(1.0 / rhoV)
            except Exception:
                P.append(np.nan); vL.append(np.nan); vV.append(np.nan)
        return np.array(P), np.array(vL), np.array(vV)
    except Exception:
        return np.array([]), np.array([]), np.array([])

# ---------- Streamlit App ----------
st.set_page_config(page_title="Steam Property Explorer", layout="wide")
st.title("Steam Property Explorer (Water)")

with st.sidebar:
    st.header("Inputs")
    fluid = "Water"
    st.markdown("**Fluid:** Water (steam)")

    st.subheader("Select Input Properties")
    input_properties = ["T", "P", "v", "x"]
    col_prop1, col_prop2 = st.columns(2)
    with col_prop1:
        input1_name = st.selectbox("Input Property 1", input_properties, index=0)
    with col_prop2:
        available_inputs2 = [prop for prop in input_properties if prop != input1_name]
        input2_name = st.selectbox("Input Property 2", available_inputs2, index=0)

    st.subheader("Enter Input Values")
    input_values = {}
    input_units = {}

    # Temperature input
    if "T" in (input1_name, input2_name):
        t_unit = st.radio("Temperature unit", ["°C", "K"], horizontal=True, key="t_unit")
        default_t = 100.0 if t_unit == "°C" else 373.15
        T_in = st.number_input(f"Temperature [{t_unit}]", value=default_t, key="T_in", format="%.3f")
        input_values["T"] = T_in
        input_units["T_unit"] = t_unit

    # Pressure input
    if "P" in (input1_name, input2_name):
        p_unit = st.radio("Pressure unit", ["kPa", "bar", "MPa", "Pa"], index=1, horizontal=True, key="p_unit")
        default_p = 1.0 if p_unit == "bar" else (101.325 if p_unit == "kPa" else (0.101325 if p_unit == "MPa" else 101325.0))
        P_in = st.number_input(f"Pressure [{p_unit}]", value=float(default_p), key="P_in", format="%.6g")
        input_values["P"] = P_in
        input_units["P_unit"] = p_unit

    # Specific volume input
    if "v" in (input1_name, input2_name):
        default_v = 0.001043
        v_in = st.number_input("Specific volume [m³/kg]", value=default_v, key="v_in", format="%.6f")
        input_values["v"] = v_in

    # Quality input
    if "x" in (input1_name, input2_name):
        if is_water(fluid):
            default_x = 0.0
            x_in = st.number_input("Quality [-]", value=default_x, min_value=0.0, max_value=1.0, step=0.01, key="x_in", format="%.4f")
            input_values["x"] = x_in
        else:
            st.warning("Quality (x) only applicable to Water.")
            input_values["x"] = None

    # Validate selection and basic presence
    validation_errors = []
    if input1_name == input2_name:
        validation_errors.append("Please select two different input properties.")
    if len([k for k in input_values.keys() if input_values.get(k) is not None]) < 2:
        validation_errors.append("Please provide values for the two selected input properties.")

    # Validate numeric ranges for supplied inputs
    # Convert to SI (safe) and check basic bounds
    def safe_convert_and_check(prop):
        val = input_values.get(prop, None)
        if val is None:
            return None, None
        try:
            if prop == "T":
                TK = to_SI_T(val, input_units.get("T_unit", "K"))
                if not np.isfinite(TK) or TK <= 0:
                    return TK, "Temperature must be > 0 K and finite."
                return TK, None
            if prop == "P":
                Pa = to_SI_P(val, input_units.get("P_unit", "Pa"))
                if not np.isfinite(Pa) or Pa <= 0:
                    return Pa, "Pressure must be > 0 Pa and finite."
                return Pa, None
            if prop == "v":
                if not np.isfinite(val) or val <= 0:
                    return None, "Specific volume must be > 0 m³/kg."
                return val, None
            if prop == "x":
                if val is None:
                    return None, None
                if not np.isfinite(val) or not (0.0 <= val <= 1.0):
                    return val, "Quality must be between 0 and 1 for water."
                return val, None
        except Exception:
            return None, f"Invalid value for {prop}."
    # apply checks
    for p in [input1_name, input2_name]:
        v_si, err = safe_convert_and_check(p)
        if err:
            validation_errors.append(err)

    inputs_valid = len(validation_errors) == 0
    if not inputs_valid:
        for msg in validation_errors:
            st.error(msg)

state = {"ok": False, "msg": "Select two properties and enter values."}

if inputs_valid:
    prop1_name = input1_name
    prop2_name = input2_name
    input1_CoolProp_name = prop1_name
    input2_CoolProp_name = prop2_name

    try:
        def get_si_value(prop):
            if prop == "T":
                return to_SI_T(input_values["T"], input_units.get("T_unit", "K"))
            elif prop == "P":
                return to_SI_P(input_values["P"], input_units.get("P_unit", "Pa"))
            elif prop == "v":
                return 1.0 / input_values["v"] if input_values["v"] != 0 else np.nan
            elif prop == "x":
                return input_values["x"]
            return None

        if prop1_name == "v":
            input1_CoolProp_name = "D"
        if prop2_name == "v":
            input2_CoolProp_name = "D"

        input1_value_SI = get_si_value(prop1_name)
        input2_value_SI = get_si_value(prop2_name)

        if input1_value_SI is None or input2_value_SI is None or np.isnan(input1_value_SI) or np.isnan(input2_value_SI):
            state = {"ok": False, "msg": "Invalid input values (conversion resulted in NaN or None)."}
        else:
            # compute properties with guarded PropsSI calls
            def cp_safe(output, i1, v1, i2, v2, fluid):
                try:
                    return PropsSI(output, i1, v1, i2, v2, fluid)
                except Exception:
                    return np.nan

            calculated_T = cp_safe("T", input1_CoolProp_name, input1_value_SI, input2_CoolProp_name, input2_value_SI, fluid)
            calculated_P = cp_safe("P", input1_CoolProp_name, input1_value_SI, input2_CoolProp_name, input2_value_SI, fluid)
            calculated_h = cp_safe("H", input1_CoolProp_name, input1_value_SI, input2_CoolProp_name, input2_value_SI, fluid)
            if np.isfinite(calculated_h):
                calculated_h = calculated_h / 1000.0
            calculated_s = cp_safe("S", input1_CoolProp_name, input1_value_SI, input2_CoolProp_name, input2_value_SI, fluid)
            if np.isfinite(calculated_s):
                calculated_s = calculated_s / 1000.0
            calculated_rho = cp_safe("D", input1_CoolProp_name, input1_value_SI, input2_CoolProp_name, input2_value_SI, fluid)
            calculated_v = 1.0 / calculated_rho if np.isfinite(calculated_rho) and calculated_rho != 0 else np.nan

            calculated_x = None
            try:
                x_val = PropsSI("Q", input1_CoolProp_name, input1_value_SI, input2_CoolProp_name, input2_value_SI, fluid)
                if math.isfinite(x_val):
                    calculated_x = x_val
            except Exception:
                calculated_x = None

            try:
                calculated_phase = PhaseSI("T", calculated_T, "P", calculated_P, fluid)
            except Exception:
                calculated_phase = None

            all_finite = all([np.isfinite(v) for v in [calculated_T, calculated_P, calculated_h, calculated_s, calculated_v]])
            state = {
                "ok": bool(all_finite),
                "msg": "" if all_finite else "One or more properties could not be calculated (check inputs).",
                "T": calculated_T,
                "P": calculated_P,
                "h": calculated_h,
                "s": calculated_s,
                "v": calculated_v,
                "x": calculated_x,
                "phase": calculated_phase,
                "input_properties": {prop1_name, prop2_name}
            }
    except Exception as e:
        state = {"ok": False, "msg": f"Error calculating properties: {e}"}

prop_col, diag_col = st.columns([0.36, 0.64])

with prop_col:
    st.subheader("State Properties")
    if state["ok"]:
        st.success("Steam state calculated successfully")
        display_t_unit = input_units.get("T_unit", "K")
        display_p_unit = input_units.get("P_unit", "Pa")
        properties_to_display = {
            "T": {"label": f"Temperature [{display_t_unit}]", "value": from_SI_T(state['T'], display_t_unit) if np.isfinite(state['T']) else None, "format": ".3f"},
            "P": {"label": f"Pressure [{display_p_unit}]", "value": from_SI_P(state['P'], display_p_unit) if np.isfinite(state['P']) else None, "format": ".6g"},
            "h": {"label": "Specific enthalpy h [kJ/kg]", "value": state['h'], "format": ".3f"},
            "s": {"label": "Specific entropy s [kJ/kg·K]", "value": state['s'], "format": ".5f"},
            "v": {"label": "Specific volume v [m³/kg]", "value": state['v'], "format": ".6f"},
            "x": {"label": "Quality [-]", "value": state['x'], "format": ".4f"},
        }
        for prop_name, prop_info in properties_to_display.items():
            if prop_name == "x":
                continue
            display_label = prop_info["label"]
            display_value = prop_info["value"]
            display_format = prop_info["format"]
            status_text = "(Input)" if prop_name in state.get("input_properties", {}) else "(Calculated)"
            if display_value is None or not np.isfinite(display_value):
                st.metric(f"{display_label} {status_text}", "N/A")
            else:
                st.metric(f"{display_label} {status_text}", f"{display_value:{display_format}}")
        # quality handling
        if state.get("x") is not None:
            display_label_x = properties_to_display["x"]["label"]
            display_value_x = properties_to_display["x"]["value"]
            display_format_x = properties_to_display["x"]["format"]
            status_text_x = "(Input)" if "x" in state.get("input_properties", {}) else "(Calculated)"
            quality_note = ""
            if not (0.0 <= display_value_x <= 1.0):
                quality_note = " (Outside 0-1 range)"
            st.metric(f"{display_label_x} {status_text_x}{quality_note}", f"{display_value_x:{display_format_x}}")
        if state.get("phase"):
            st.caption(f"Phase (CoolProp): {state['phase']}")
    else:
        st.error(state["msg"])

with diag_col:
    st.subheader("Diagrams")
    tabs = st.tabs(["P–v", "T–s", "P–h"])
    # --- P–v ---
    with tabs[0]:
        try:
            fig, ax = plt.subplots()
            P_dome, vL_dome, vV_dome = sat_curves_Pv_water()
            Tc, Pc, vc = water_crit()
            if P_dome.size and vL_dome.size:
                ax.plot(vL_dome, P_dome, lw=1.3, label="Sat. liquid")
                ax.plot(vV_dome, P_dome, lw=1.3, label="Sat. vapour")
                if np.isfinite(vc) and np.isfinite(Pc):
                    ax.scatter([vc], [Pc], marker="*", s=80, label="Critical point")
            if state["ok"] and state.get("T") is not None and np.isfinite(state["T"]):
                T_si = state["T"]
                v_iso = np.geomspace(1e-4, 10.0, 220)
                P_iso = []
                for v_ in v_iso:
                    rho_ = 1.0 / v_
                    try:
                        P_iso.append(PropsSI("P", "T", T_si, "D", rho_, fluid))
                    except Exception:
                        P_iso.append(np.nan)
                ax.plot(v_iso, P_iso, alpha=0.55, lw=0.9, label=f"Isotherm @ {T_si:.1f} K")
            if state["ok"] and state.get("v") is not None and state.get("P") is not None:
                ax.scatter([state["v"]], [state["P"]], marker="o", label="Current state")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("v [m³/kg]"); ax.set_ylabel("P [Pa]")
            ax.grid(True, which="both", ls=":")
            ax.legend(loc="best", fontsize=8)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"P–v plot issue: {e}")
    # --- T–s ---
    with tabs[1]:
        try:
            fig, ax = plt.subplots()
            Tgrid, sL, sV = sat_curves_Ts_water()
            if Tgrid.size:
                ax.plot(sL, Tgrid, lw=1.3, label="Sat. liquid")
                ax.plot(sV, Tgrid, lw=1.3, label="Sat. vapour")
            try:
                Tc, Pc, _ = water_crit()
                if np.isfinite(Tc) and np.isfinite(Pc):
                    sc = PropsSI("S", "T", Tc, "P", Pc, "Water") / 1000.0
                    ax.scatter([sc], [Tc], marker="*", s=80, label="Critical point")
            except Exception:
                pass
            pressures_to_plot = [1e4, 1e5, 1e6, 1e7]
            for P_const in pressures_to_plot:
                try:
                    T_triple, _ = water_triple()
                    T_vals = np.linspace(max(T_triple, 1e-6), 1000.0, 200)
                    s_vals = []
                    valid_T = []
                    for T_val in T_vals:
                        try:
                            s_val = PropsSI("S", "T", T_val, "P", P_const, fluid) / 1000.0
                            s_vals.append(s_val); valid_T.append(T_val)
                        except Exception:
                            s_vals.append(np.nan); valid_T.append(np.nan)
                    valid_s_vals = np.array(s_vals)[np.isfinite(valid_T)]
                    valid_T_vals = np.array(valid_T)[np.isfinite(valid_T)]
                    if len(valid_s_vals) > 1:
                        ax.plot(valid_s_vals, valid_T_vals, alpha=0.7, lw=0.8, ls='--', label=f'{from_SI_P(P_const, "bar"):.2g} bar')
                except Exception:
                    continue
            if state["ok"] and state.get("s") is not None and state.get("T") is not None:
                ax.scatter([state["s"]], [state["T"]], marker="o", label="Current state")
            ax.set_xlabel("s [kJ/kg·K]"); ax.set_ylabel("T [K]")
            ax.grid(True, ls=":"); ax.legend(loc="best", fontsize=8)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"T–s plot issue: {e}")
# ...existing code...
    # --- P–h ---
    with tabs[2]:
        try:
            fig, ax = plt.subplots()
            Pgrid, hL, hV = sat_curves_Ph_water()
            if Pgrid.size:
                ax.plot(hL, Pgrid, lw=1.3, label="Sat. liquid")
                ax.plot(hV, Pgrid, lw=1.3, label="Sat. vapour")
            try:
                Tc, Pc, _ = water_crit()
                if np.isfinite(Tc) and np.isfinite(Pc):
                    hc = PropsSI("H", "T", Tc, "P", Pc, "Water") / 1000.0
                    ax.scatter([hc], [Pc], marker="*", s=80, label="Critical point")
            except Exception:
                pass
            volumes_to_plot = [1e-3, 1e-2, 1e-1, 1, 10]
            for v_const in volumes_to_plot:
                try:
                    rho_const = 1.0 / v_const
                    P_vals = np.geomspace(1e3, 1e8, 200)
                    h_vals = []
                    for P_val in P_vals:
                        try:
                            h_val = PropsSI("H", "P", P_val, "D", rho_const, fluid) / 1000.0
                            h_vals.append(h_val)
                        except Exception:
                            h_vals.append(np.nan)
                    valid_h_vals = np.array(h_vals)[np.isfinite(P_vals)]
                    valid_P_vals = np.array(P_vals)[np.isfinite(P_vals)]
                    if len(valid_h_vals) > 1:
                        ax.plot(valid_h_vals, valid_P_vals, alpha=0.7, lw=0.8, ls='--', label=f'v={v_const:.0e} m³/kg')
                except Exception:
                    continue
            if state["ok"] and state.get("h") is not None and state.get("P") is not None:
                ax.scatter([state["h"]], [state["P"]], marker="o", label="Current state")
            ax.set_yscale("log")
            ax.set_xlabel("h [kJ/kg]"); ax.set_ylabel("P [Pa]")
            # Limit x-axis to 5000
            ax.set_xlim(left=0, right=5000)
            ax.grid(True, which="both", ls=":"); ax.legend(loc="best", fontsize=8)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"P–h plot issue: {e}")

st.caption("Plots update live. Saturation domes & critical point shown for Water.")
# ...existing code...