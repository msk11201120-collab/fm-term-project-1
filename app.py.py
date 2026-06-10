import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 페이지 기본 설정 및 레이아웃
st.set_page_config(page_title="PR Dispensing Jet Simulator", layout="wide")

st.title("🔬 Photoresist Dispensing Jet: Rayleigh-Plateau Instability Simulator")
st.markdown("---")

# 固定된 물리 상수 (포토레지스트 밀도 및 공기 밀도 예시값)
RHO_PR = 1000.0  # kg/m^3
RHO_AIR = 1.2    # kg/m^3

# ==========================================
# 2. 사이드바: 입력 파라미터 제어 (Sliders)
# ==========================================
st.sidebar.header("🎛️ Process & Equipment Specs")

# 노즐 기하구조 및 수송 거리
r0 = st.sidebar.slider("Nozzle Radius (r₀, μm)", min_value=10, max_value=200, value=50, step=5) * 1e-6
L = st.sidebar.slider("Distance to Substrate (L, mm)", min_value=1, max_value=30, value=10, step=1) * 1e-3

# 운전 조건 및 물성치
U = st.sidebar.slider("Ejection Velocity (U, m/s)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
eta = st.sidebar.slider("PR Dynamic Viscosity (η, mPa·s)", min_value=1, max_value=100, value=20, step=1) * 1e-3
gamma = st.sidebar.slider("Surface Tension (γ, mN/m)", min_value=10, max_value=70, value=30, step=1) * 1e-3

# 섭동 초기값 설정 (Breakup Time 계산용)
eps_ratio = st.sidebar.slider("Initial Perturbation Ratio (ε₀/r₀)", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.1e")

# ==========================================
# 3. 메인 화면: 무차원수 대시보드 표시 (Consistency Score)
# ==========================================
# 무차원수 계산
Re = (RHO_PR * U * (2 * r0)) / eta
We = (RHO_PR * (U**2) * r0) / gamma
Ca = (eta * U) / gamma
Oh = eta / np.sqrt(RHO_PR * gamma * r0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Reynolds Number (Re)", f"{Re:.2f}")
col2.metric("Weber Number (We)", f"{We:.2f}")
col3.metric("Capillary Number (Ca)", f"{Ca:.4f}")
col4.metric("Ohnesorge Number (Oh)", f"{Oh:.4f}")

st.markdown("---")

# ==========================================
# 4. 물리 계산 엔진 (Physics Engine)
# ==========================================
# 무차원 파수 x = k * r0 범위 설정 (0에서 1까지만 불안정 영역)
x = np.linspace(0.01, 1.5, 200)

# 1) Inviscid Limit (Rayleigh 고전 이론 해)
omega_inviscid = np.sqrt((gamma / (RHO_PR * r0**3)) * x * (1 - x**2))

# 2) Viscous Limit (Weber 이론 반영한 실제 성장률)
# 대략적인 점성 감쇄 공식 적용 (오메가 스펙트럼 유도)
omega_viscous = np.sqrt((gamma / (RHO_PR * r0**3)) * x * (1 - x**2) + (3 * eta * x**2 / (RHO_PR * r0**2))**2) - (3 * eta * x**2 / (RHO_PR * r0**2))
omega_viscous = np.nan_to_num(omega_viscous) # 음수나 에러값 방지

# 가장 빠르게 성장하는 지점 도출
max_idx = np.argmax(omega_viscous)
x_max = x[max_idx]
omega_max = omega_viscous[max_idx]

# 지배적 파장, 파단 시간, 파단 거리 계산
k_max = x_max / r0
dominant_wavelength = (2 * np.pi) / k_max
t_b = (1 / omega_max) * np.log(1 / eps_ratio) if omega_max > 0 else float('inf')
L_b = U * t_b

# ==========================================
# 5. 메인 레이아웃 탭 분할 (교수님 요구 뷰 매핑)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Validation View", "🌊 Core Simulation & Animation", "🎯 Design-exploration Mode"])

# ------------------------------------------
# Tab 1: Validation View (해석해와 비교 검증)
# ------------------------------------------
with tab1:
    st.subheader("Instability Growth-Rate Spectrum: Model Validation")
    st.markdown("Comparing current viscous PR jet simulation with classic **Inviscid Limit (Rayleigh Limit)**.")
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, omega_inviscid, 'r--', label='Inviscid Limit (Rayleigh)')
    ax.plot(x, omega_viscous, 'b-', linewidth=2, label='Current PR Jet (Viscous)')
    ax.axvline(x=1.0, color='gray', linestyle=':', label='Cutoff Wavenumber (x=1)')
    ax.scatter(x_max, omega_max, color='black', zorder=5, label=f'Max Growth (x={x_max:.3f})')
    
    ax.set_xlabel("Dimensionless Wavenumber ($x = k r_0$)")
    ax.set_ylabel("Growth Rate ($\omega$, 1/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # 정량적 결과 리포트 대시보드
    st.markdown("### 📋 Predictive Breakup Outputs")
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Dominant Breakup Wavelength ($\lambda_{{max}}$):** {dominant_wavelength*1e6:.1f} $\mu$m")
    c2.write(f"**Estimated Breakup Time ($t_b$):** {t_b*1e3:.3f} ms")
    if L_b < L:
        c3.markdown(f"**Breakup Distance ($L_b$):** <span style='color:red; font-weight:bold;'>{L_b*1e3:.2f} mm</span> (Breaks before wafer!)", unsafe_allow_html=True)
    else:
        c3.markdown(f"**Breakup Distance ($L_b$):** <span style='color:green; font-weight:bold;'>{L_b*1e3:.2f} mm</span> (Stable transport!)", unsafe_allow_html=True)

# ------------------------------------------
# Tab 2: Core Animation (실시간 파형 시각화)
# ------------------------------------------
with tab2:
    st.subheader("Liquid Column Waveform Deformation")
    
    # 간단한 정적/동적 형태 시각화 (수송 거리 L 기준)
    z_plot = np.linspace(0, L, 500)
    
    # 거리에 따른 섭동 진폭 성장 계산
    # z = U * t 이므로 t = z / U
    r_profile = []
    for zi in z_plot:
        if zi < L_b:
            # 파단 전: 사인파 형태로 진폭 증가
            amp = r0 * eps_ratio * np.exp(omega_max * (zi / U))
            r_val = r0 + amp * np.sin(k_max * zi)
        else:
            # 파단 후: 액적으로 쪼개짐 연출 (0 혹은 불연속 점)
            r_val = 0.0 if np.sin(k_max * zi) < 0 else r0 * 1.5
        r_profile.append(r_val)
        
    r_profile = np.array(r_profile)
    
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(z_plot * 1e3, r_profile * 1e6, 'teal', label='Upper Interface')
    ax2.plot(z_plot * 1e3, -r_profile * 1e6, 'teal', label='Lower Interface')
    ax2.axvline(x=L*1e3, color='darkred', linestyle='-', label='Wafer Surface')
    if L_b < L:
        ax2.axvline(x=L_b*1e3, color='orange', linestyle='--', label='Breakup Point')
        
    ax2.set_xlabel("Transport Distance (z, mm)")
    ax2.set_ylabel("Jet Radius ($r, \mu$m)")
    ax2.set_ylim([-r0*2.5*1e6, r0*2.5*1e6])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)
    st.pyplot(fig2)
    
    # 성공/실패 경고 메시지
    if L_b < L:
        st.error(f"🚨 **PROCESS DEFECT DETECTED:** 제트가 웨이퍼 기판에 도달하기 전인 {L_b*1e3:.2f} mm 지점에서 쪼개집니다. 두께 불균일 및 기포 포집 불량이 발생할 수 있습니다!")
    else:
        st.success("✅ **STABLE TRANSPORT:** 제트가 파단 없이 매끄러운 기둥 형태로 웨이퍼 표면에 안전하게 충돌합니다.")

# ------------------------------------------
# Tab 3: Design-exploration Mode (엔지니어 가이드)
# ------------------------------------------
with tab3:
    st.subheader("Fab Engineer's Operational Window Guide")
    st.markdown("This map shows the **Stable vs. Unstable** regions based on Velocity and Distance under current liquid properties.")
    
    # 2D 맵 그리드 생성 (속도 vs 수송거리)
    u_range = np.linspace(0.5, 6.0, 100)
    l_range = np.linspace(1, 25, 100)
    
    # 현재 설정 조건의 점 하나 표시용
    current_u = U
    current_l = L * 1e3
    
    # 경계선 계산: L_b = U * t_b -> 각 U마다 파단이 일어나는 임계 거리 L_boundary 계산
    # 오메가 최대값은 속도와 무관하므로 t_b는 고정됨 (베르누이 속도와 별개로 유체 고유 불안정성 기준)
    l_boundary = u_range * t_b * 1e3
    
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.plot(u_range, l_boundary, 'k-', linewidth=2, label='Breakup Boundary ($L_b = U \cdot t_b$)')
    ax3.fill_between(u_range, 0, l_boundary, color='red', alpha=0.15, label='Defect Zone (Premature Breakup)')
    ax3.fill_between(u_range, l_boundary, max(l_range), color='green', alpha=0.15, label='Safe Zone (Continuous Jet)')
    
    # 현재 작동 상태 플롯
    ax3.scatter(current_u, current_l, color='purple', s=120, edgecolors='black', zorder=10, label='Current Operating Point')
    
    ax3.set_xlabel("Ejection Velocity (U, m/s)")
    ax3.set_ylabel("Nozzle-to-Wafer Distance (L, mm)")
    ax3.set_xlim([min(u_range), max(u_range)])
    ax3.set_ylim([0, max(l_range)])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    
    # 엔지니어 처방전 제안
    st.markdown("### 💡 Process Design Recommendation")
    if L_b < L:
        st.warning(f"현재 조건은 **Defect Zone**에 있습니다. 안전 영역(Safe Zone)으로 이동하려면 **1) 노즐 압력을 높여 유속(U)을 증가시키거나, 2) 노즐과 기판 사이의 간격(L)을 줄이십시오.**")
    else:
        st.info("현재 조건은 **Safe Zone**에 안전하게 배치되어 있습니다. 공정 마진 확보를 위해 점도 점검을 정기적으로 유지하십시오.")