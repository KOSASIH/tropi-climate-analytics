"""
water_quality_monitor.py — Sprint 11 P1
Remote-sensing WQI for 20 Indonesian lakes/reservoirs.
MODIS/Landsat band ratios + KepMenLH-115/2003 composite.
"""
from __future__ import annotations
import json, logging, math, os, random
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_WS          = Path(os.environ.get("HYDRO_WORKSPACE", "/opt/airflow/workspace"))
_WQ_DIR      = _WS / "output" / "water_quality"
_MODIS_DIR   = _WS / "data"   / "modis"
_LANDSAT_DIR = _WS / "data"   / "landsat"
_CFG_PATH    = _WS / "config" / "water_bodies_config.json"

_CHLA_OLIGO=2.0; _CHLA_MESO=10.0; _CHLA_HYPER=50.0
_TURB_LOW=10.0;  _TURB_MOD=50.0;  _TURB_HIGH=200.0
_WQI_EXCELLENT=90; _WQI_GOOD=70; _WQI_FAIR=50; _WQI_POOR=25


@dataclass
class WQIResult:
    water_body_id: str; date: date; wqi: float; category: str
    turbidity_ntu: float; chlorophyll_ug_l: float; cdom_ratio: float
    surface_temp_c: float; temp_anomaly_c: float; pixel_count: int
    cloud_coverage_pct: float; data_source: str; eutrophication_risk: str

@dataclass
class TurbidityMap:
    region_id: str; date: date; mean_turbidity_ntu: float
    high_turbidity_pct: float; spatial_grid: list = field(default_factory=list)

@dataclass
class ChlorophyllResult:
    water_body_id: str; date: date; chlorophyll_ug_l: float
    oc3m_rrs_ratio: float; data_source: str; confidence: str

@dataclass
class EutrophicationRisk:
    water_body_id: str; date: date; risk_level: str
    chlorophyll_ug_l: float; total_phosphorus_proxy: float
    swir_ratio: float; bloom_probability: float


class WaterQualityMonitor:
    """MODIS/Landsat water quality indexing for 20 Indonesian water bodies."""

    def __init__(self) -> None:
        self._cfg = json.loads(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}

    def compute_wqi(self, water_body_id: str, dt: date) -> WQIResult:
        rs  = self._load_rs(water_body_id, dt)
        t   = self._turbidity(rs); c = self._chlorophyll(rs)
        cdom = self._cdom(rs); temp = self._lst(rs)
        ta   = temp - {"danau_toba":23.5,"danau_singkarak":25.0,"danau_maninjau":26.5,
                        "danau_towuti":27.0,"danau_matano":26.8}.get(water_body_id, 27.5)
        wqi  = (self._s_turb(t)*0.40 + self._s_chla(c)*0.35 +
                self._s_cdom(cdom)*0.15 + self._s_ta(abs(ta))*0.10)
        cat  = self._cat(wqi); eutro = self._eutro(c)
        r    = WQIResult(water_body_id, dt, round(wqi,2), cat,
                         round(t,3), round(c,3), round(cdom,4),
                         round(temp,2), round(ta,2),
                         rs.get("pixel_count",0), round(rs.get("cloud_coverage_pct",0.0),1),
                         rs.get("source","synthetic"), eutro)
        self._emit(water_body_id, cat, wqi); self._write(r)
        return r

    def compute_spatial_turbidity(self, region_id: str, dt: date) -> TurbidityMap:
        rs = self._load_rs(region_id, dt); mt = self._turbidity(rs)
        grid = [{"lat": dlat, "lon": dlon,
                 "turbidity_ntu": round(max(0.0, mt+random.gauss(0, mt*0.15)),2)}
                for dlat in (-0.1,0.0,0.1) for dlon in (-0.1,0.0,0.1)]
        hp = sum(1 for g in grid if g["turbidity_ntu"]>_TURB_MOD)/len(grid)*100
        return TurbidityMap(region_id, dt, round(mt,3), round(hp,1), grid)

    def compute_chlorophyll(self, water_body_id: str, dt: date) -> ChlorophyllResult:
        rs  = self._load_rs(water_body_id, dt); c = self._chlorophyll(rs)
        conf = "high" if rs.get("source")=="landsat" else ("medium" if rs.get("source")=="modis" else "low")
        return ChlorophyllResult(water_body_id, dt, round(c,3), round(c/10.0,4), rs.get("source","synthetic"), conf)

    def get_eutrophication_risk(self, water_body_id: str, wqi: WQIResult) -> EutrophicationRisk:
        rs   = self._load_rs(water_body_id, wqi.date)
        sw   = rs.get("swir_ratio",0.05)
        bp   = min(1.0, (wqi.chlorophyll_ug_l/_CHLA_HYPER)**0.8)
        return EutrophicationRisk(water_body_id, wqi.date, self._eutro(wqi.chlorophyll_ug_l),
                                   wqi.chlorophyll_ug_l, round(sw*180,2), round(sw,4), round(bp,3))

    def update_active_wqi(self, results: list) -> dict:
        _WQ_DIR.mkdir(parents=True, exist_ok=True)
        _rank={"excellent":0,"good":1,"fair":2,"poor":3,"very_poor":4}
        hc = max(results, key=lambda r:_rank.get(r.category,0)).category if results else "excellent"
        sc = {"generated_at":datetime.now(tz=timezone.utc).isoformat(),
              "water_body_count":len(results),
              "highest_concern_category":hc,
              "poor_count":sum(1 for r in results if r.category in("poor","very_poor")),
              "results":[self._rdict(r) for r in results]}
        (_WQ_DIR/"active_wqi.json").write_text(json.dumps(sc,indent=2,default=str))
        return sc

    # ── RS math ──────────────────────────────────────────────────────────────

    def _turbidity(self,rs):
        r=max(rs.get("rho_red",0.05)/max(rs.get("rho_nir",0.03),1e-6),0.01)
        return 10**(1.1317*math.log(r)+1.0848)

    def _chlorophyll(self,rs):
        r=max(rs.get("rrs443",0.008)/max(rs.get("rrs555",0.012),1e-6),0.01)
        return max(0.0,10**(0.2424-2.7423*math.log10(r)))

    def _cdom(self,rs):
        return rs.get("rrs412",0.006)/max(rs.get("rrs555",0.012),1e-6)

    def _lst(self,rs): return rs.get("lst_celsius",28.5+random.gauss(0,1.5))

    def _s_turb(self,n):
        if n<_TURB_LOW: return 100.
        if n<_TURB_MOD: return 100-(n-_TURB_LOW)/(_TURB_MOD-_TURB_LOW)*50
        if n<_TURB_HIGH:return 50 -(n-_TURB_MOD)/(_TURB_HIGH-_TURB_MOD)*25
        return max(0.,25-(n-_TURB_HIGH)/200*25)

    def _s_chla(self,c):
        if c<_CHLA_OLIGO: return 100.
        if c<_CHLA_MESO:  return 100-(c-_CHLA_OLIGO)/(_CHLA_MESO-_CHLA_OLIGO)*40
        if c<_CHLA_HYPER: return 60 -(c-_CHLA_MESO)/(_CHLA_HYPER-_CHLA_MESO)*35
        return max(0.,25-(c-_CHLA_HYPER)/100*25)

    def _s_cdom(self,v): return max(0.,100-(v-1.0)*50) if v>1.0 else 100.

    def _s_ta(self,d):
        if d<1:return 100.
        if d<2:return 100-(d-1)*40
        if d<4:return 60-(d-2)*20
        return max(0.,20-(d-4)*5)

    @staticmethod
    def _cat(w):
        if w>=_WQI_EXCELLENT:return "excellent"
        if w>=_WQI_GOOD:return "good"
        if w>=_WQI_FAIR:return "fair"
        if w>=_WQI_POOR:return "poor"
        return "very_poor"

    @staticmethod
    def _eutro(c):
        if c<_CHLA_OLIGO:return "OLIGOTROPHIC"
        if c<_CHLA_MESO: return "MESOTROPHIC"
        if c<_CHLA_HYPER:return "EUTROPHIC"
        return "HYPEREUTROPHIC"

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load_rs(self, wbid: str, dt: date) -> dict:
        for path, src in [
            (_LANDSAT_DIR/f"L8_OLI_{wbid}_{dt.strftime('%Y%m%d')}.json","landsat"),
            (_MODIS_DIR/f"MOD09GA_{wbid}_{dt.strftime('%Y%m%d')}.json","modis"),
        ]:
            if path.exists():
                try:
                    d=json.loads(path.read_text())
                    if src=="landsat" and d.get("cloud_coverage_pct",100)>=20: continue
                    d["source"]=src; return d
                except Exception: pass
        random.seed(sum(ord(c) for c in wbid)+dt.toordinal())
        return {"source":"synthetic","rho_red":random.uniform(0.02,0.12),
                "rho_nir":random.uniform(0.02,0.06),"rrs443":random.uniform(0.004,0.015),
                "rrs555":random.uniform(0.008,0.020),"rrs412":random.uniform(0.003,0.010),
                "swir_ratio":random.uniform(0.02,0.10),"lst_celsius":random.uniform(25.0,32.0),
                "cloud_coverage_pct":random.uniform(10,40),"pixel_count":random.randint(500,5000)}

    def _write(self,r):
        _WQ_DIR.mkdir(parents=True,exist_ok=True)
        dt_str = r.date.strftime('%Y%m%d') if hasattr(r.date,'strftime') else str(r.date).replace('-','')
        (_WQ_DIR/f"wqi_{r.water_body_id}_{dt_str}.json").write_text(
            json.dumps(self._rdict(r),indent=2,default=str))

    @staticmethod
    def _rdict(r):
        return {"water_body_id":r.water_body_id,
                "date":r.date.isoformat() if hasattr(r.date,"isoformat") else str(r.date),
                "wqi":r.wqi,"category":r.category,"turbidity_ntu":r.turbidity_ntu,
                "chlorophyll_ug_l":r.chlorophyll_ug_l,"cdom_ratio":r.cdom_ratio,
                "surface_temp_c":r.surface_temp_c,"temp_anomaly_c":r.temp_anomaly_c,
                "pixel_count":r.pixel_count,"cloud_coverage_pct":r.cloud_coverage_pct,
                "data_source":r.data_source,"eutrophication_risk":r.eutrophication_risk}

    @staticmethod
    def _emit(wbid,cat,wqi):
        try:
            from src.hydrology.metrics import WATER_QUALITY_INDEX
            WATER_QUALITY_INDEX.labels(water_body_id=wbid,category=cat).set(wqi)
        except Exception as e: logger.debug("WQI metric: %s",e)
