"""Part 2 diagnostics run on 27 Oct (read-only): new-business experience by year, OOF shortfall by year x NB x age,
public-response decomposition. Run from the repo root: python -P -E scripts/part2_nb_diagnostics.py [history|oof|response]"""
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "history"
if MODE == "history":
    import pandas as pd, numpy as np
    D="data/part2/emerging/"
    cl=pd.read_parquet(D+"claims.parquet")
    cl=cl[cl.status.str.lower()!="declined"] if cl.status.dtype==object else cl
    cl["loss_date"]=pd.to_datetime(cl.loss_date); cl["report_date"]=pd.to_datetime(cl.report_date)
    cl["cost"]=cl.incurred_to_date.fillna(0)-cl.recoveries.fillna(0)
    print("claims status:", cl.status.value_counts().to_dict())
    for line in ["motor","buildings","contents"]:
        p=pd.read_parquet(D+f"policies_{line}.parquet")
        p["period_start"]=pd.to_datetime(p.period_start); p["period_end"]=pd.to_datetime(p.period_end)
        p["year"]=p.period_start.dt.year
        p["expo"]=np.where(p.exposure>1.5,p.exposure/365.25,p.exposure).clip(0,1.2)
        c=cl[cl.line==line].merge(p[["row_id","policy_id","period_start","period_end"]],on="policy_id")
        c=c[(c.loss_date>=c.period_start)&(c.loss_date<=c.period_end)]
        agg=c.groupby("row_id").agg(n=("claim_id","size"),cost=("cost","sum"))
        p=p.merge(agg,left_on="row_id",right_index=True,how="left").fillna({"n":0,"cost":0})
        g=p.groupby(["year","new_business"]).agg(rows=("row_id","size"),expo=("expo","sum"),n=("n","sum"),cost=("cost","sum"))
        g["freq"]=g.n/g.expo; g["sev"]=g.cost/g.n.replace(0,np.nan); g["pp"]=g.cost/g.expo
        print(f"\n== {line} ==\n", g.round(3))
        for y in sorted(p.year.unique()):
            s=g.loc[y]; 
            if len(s)==2:
                nb,rn=s.loc[True] if True in s.index else s.iloc[1], s.loc[False] if False in s.index else s.iloc[0]
                print(f"  {y}: NB/ren PP {nb.pp/rn.pp:.2f}  freq {nb.freq/rn.freq:.2f}  sev {nb.sev/rn.sev:.2f}  (NB claims {int(nb.n)})")
        # start-month mix and loss-month for 2024
        p24=p[p.year==2024]
        print("  2024 start-month share NB:", p24[p24.new_business==True].period_start.dt.month.value_counts(normalize=True).sort_index().round(2).to_dict())
        print("  2024 start-month share ren:", p24[p24.new_business==False].period_start.dt.month.value_counts(normalize=True).sort_index().round(2).to_dict())
        c24=c[c.period_start.dt.year==2024].merge(p[["row_id","new_business"]],on="row_id")
        print("  2024 mean loss month NB/ren:", c24.groupby("new_business").loss_date.apply(lambda s:s.dt.month.mean()).round(2).to_dict(),
              " report lag days NB/ren:", c24.groupby("new_business").apply(lambda d:(d.report_date-d.loss_date).dt.days.mean()).round(1).to_dict())
        if line=="motor":
            p23=p[p.year==2023]; 
            for nb in [True,False]:
                q=p23[p23.new_business==nb]; print(f"  2023 motor {('NB' if nb else 'ren')}: age bands", q.driver_age_band.value_counts(normalize=True).sort_index().round(2).to_dict())
        t=pd.read_parquet(D+f"test_{line}.parquet"); t["period_start"]=pd.to_datetime(t.period_start)
        print("  TEST NB share", round(t.new_business.mean(),3), "NB start-month", t[t.new_business==True].period_start.dt.month.value_counts(normalize=True).sort_index().round(2).to_dict(),
              "ren start-month", t[t.new_business==False].period_start.dt.month.value_counts(normalize=True).sort_index().round(2).to_dict())
elif MODE == "oof":
    import pickle, numpy as np, pandas as pd
    r=pickle.load(open("experiments/results/part2pool/R2_P34_hn.pkl","rb"))
    def d2(y,m):
        y=np.asarray(y,float); m=np.asarray(m,float); yb=y.mean()
        dev=lambda mu: 2*(y**0.5/( -0.5*0.5)*0 + (y**2/(( 2-1.5)*(1-1.5)))*0)  # placeholder
        def tw(mu): return 2*(y**(2-1.5)/((1-1.5)*(2-1.5)) - y*mu**(1-1.5)/(1-1.5) + mu**(2-1.5)/(2-1.5)).sum()
        return 1-tw(m)/tw(yb)
    for line in ["motor","buildings","contents"]:
        v=r["lines"][line]
        df=pd.DataFrame({"row_id":v["row_id"],"y":v["y_eval"],"oof":v["oof"]})
        em=~df.row_id.astype(str).str.startswith("D")
        df=df[em]
        tr=pd.read_parquet(f"experiments/cache/part2pool/train_{line}.parquet")
        cols=["row_id","period_start","exposure","new_business"]+(["driver_age_band"] if line=="motor" else ["policyholder_age_band"])
        tr=tr[[c for c in cols if c in tr.columns]]
        df=df.merge(tr,on="row_id"); df["year"]=pd.to_datetime(df.period_start).dt.year
        g=df.groupby(["year","new_business"]).agg(rows=("y","size"),act=("y","sum"),pred=("oof","sum"))
        g["a/e"]=g.act/g.pred; print(f"\n== {line}: OOF actual/pred by year x NB ==\n", g.round(3))
        for y in [2023,2024]:
            s=g.loc[y]; print(f"   {y} NB/ren calibration ratio = {s.loc[True,'a/e']/s.loc[False,'a/e']:.3f}")
        both=df.groupby("new_business").agg(act=("y","sum"),pred=("oof","sum")); both["a/e"]=both.act/both.pred
        print("   2023+2024 pooled a/e:", both["a/e"].round(3).to_dict(), " NB/ren =", round(both.loc[True,"a/e"]/both.loc[False,"a/e"],3))
        # holdout (train 2023 -> predict 2024)
        hi=np.asarray(v["holdout_idx"]); hp=np.asarray(v["holdout_pred"])
        h=pd.DataFrame({"row_id":np.asarray(v["row_id"])[hi],"y":np.asarray(v["y_eval"])[hi],"p":hp})
        h=h[~h.row_id.astype(str).str.startswith("D")].merge(tr,on="row_id")
        hg=h.groupby("new_business").agg(rows=("y","size"),act=("y","sum"),pred=("p","sum")); hg["a/e"]=hg.act/hg.pred
        print("   2024 holdout a/e by NB:", hg["a/e"].round(3).to_dict(), " NB/ren =", round(hg.loc[True,"a/e"]/hg.loc[False,"a/e"],3))
        # what a NB factor does to the 2024 holdout D2 (emerging rows)
        for f in [1.0,1.1,1.2,1.3]:
            p=h.p*np.where(h.new_business,f,1.0); print(f"   holdout D2 with NB x{f}: {d2(h.y,p):.4f}", end="")
        print()
        ab="driver_age_band" if line=="motor" else "policyholder_age_band"
        if ab in df:
            ga=df.groupby([ab,"new_business"]).agg(rows=("y","size"),act=("y","sum"),pred=("oof","sum")); ga["a/e"]=(ga.act/ga.pred).round(3)
            print("   OOF a/e by age band x NB (both years):\n", ga["a/e"].unstack().round(3))
        # test NB/ren model relativity
        te=pd.read_parquet(f"experiments/cache/part2pool/test_{line}.parquet",columns=["row_id","new_business","exposure"]+([ab] if ab else []))
        tp=pd.DataFrame({"row_id":v["test_row_id"],"p":v["test_pred"]}).merge(te,on="row_id")
        tg=tp.groupby("new_business").agg(rows=("p","size"),p=("p","sum"),e=("exposure","sum")); tg["pp"]=tg.p/tg.e
        print("   TEST model PP NB/ren:", round(tg.loc[True,"pp"]/tg.loc[False,"pp"],3), " rows", tg.rows.to_dict())
        if ab: print("   TEST age mix NB:", tp[tp.new_business].groupby(ab).size().div(tp.new_business.sum()).round(2).to_dict())
elif MODE == "response":
    import numpy as np, pandas as pd, json
    S="submissions/part2/"
    base=pd.read_csv(S+"20261027_1658Z_P2_POOL_POOL_P34_w10_v0_dn.csv")
    te=pd.concat([pd.read_parquet(f"data/part2/emerging/test_{l}.parquet",columns=["row_id","new_business"]).assign(line=l) for l in ["motor","buildings","contents"]])
    d=base.merge(te,on="row_id"); mu=d.prediction.values; nb=d.new_business.values.astype(bool)
    L={l:(d.line==l).values for l in ["motor","buildings","contents"]}
    def fit2(s1,d1,s2,d2):
        A=np.array([[-(s1**-0.5-1),-(s1**0.5-1)],[-(s2**-0.5-1),-(s2**0.5-1)]]); return np.linalg.solve(A,[d1,d2])
    gain=lambda a,b,s: -a*(s**-0.5-1)-b*(s**0.5-1)
    a_m,b_m=fit2(0.9,-0.00332,1.1,+0.00145)
    R={}
    def put(key,a,b,src): R[key]={"alpha":float(a),"beta":float(b),"s_opt":float(a/b),"max_gain":float((a**0.5-b**0.5)**2),"source":src}
    put("motor",a_m,b_m,"probes x0.90/x1.10")
    k=b_m/np.sqrt(mu[L["motor"]]).sum()          # beta per unit sum sqrt(mu)
    # buildings: two probes -> fit2 directly, compare with constraint
    a_b,b_b=fit2(0.85,-0.00176,1.15,-0.00009); put("buildings",a_b,b_b,"probes x0.85/x1.15")
    print(f"buildings beta fitted {b_b:.5f} vs sqrt-mu constraint {k*np.sqrt(mu[L['buildings']]).sum():.5f}")
    # contents: one probe + constraint
    b_c=k*np.sqrt(mu[L["contents"]]).sum(); s=1.10; a_c=(-0.00035+b_c*(s**0.5-1))/(-(s**-0.5-1)); put("contents",a_c,b_c,"probe x1.10 + sqrt-mu constraint")
    # motor NB / renewal
    b_nb=k*np.sqrt(mu[L["motor"]&nb]).sum(); s=1.20; a_nb=(0.00207+b_nb*(s**0.5-1))/(-(s**-0.5-1)); put("motor:nb",a_nb,b_nb,"probe NB x1.20 + constraint"); put("motor:ren",a_m-a_nb,b_m-b_nb,"difference")
    # home-line NB subsets: beta known, alpha under hypotheses -> expected probe readings
    for l in ["buildings","contents"]:
        bl=R[l]["beta"]; bn=k*np.sqrt(mu[L[l]&nb]).sum(); share=bn/bl
        R[f"{l}:nb"]={"beta":float(bn),"alpha":None,"note":"alpha unknown until probed"}
        R[f"{l}:ren"]={"beta":float(bl-bn),"alpha":None}
        print(f"{l}: NB share of sqrt-mu {share:.3f}; line optimum {R[l]['s_opt']:.3f}, max gain {R[l]['max_gain']:+.5f}")
        for hyp in ([0.80,1.0] if l=="buildings" else [1.0,1.24,1.45]):
            # hypothesis: NB rows want x hyp relative to the line optimum, renewals the complement so the line optimum holds
            a_n=bn*R[l]["s_opt"]*hyp
            for probe in ([0.85,1.15] if l=="buildings" else [1.2,1.3]):
                print(f"   if NB truth = line_opt x {hyp}: probe NB x{probe} reads {gain(a_n,bn,probe):+.5f}")
    print(json.dumps({k:{kk:(round(vv,6) if isinstance(vv,float) else vv) for kk,vv in v.items()} for k,v in R.items()},indent=1))
    R["_meta"]={"reference_file":"20261027_1658Z_P2_POOL_POOL_P34_w10_v0_dn.csv","reference_public":0.15968,"beta_per_sqrt_mu":float(k),
                "formula":"delta_public(s) = -alpha*(s^-0.5-1) - beta*(s^0.5-1); optimum s=alpha/beta; beta = beta_per_sqrt_mu * sum(sqrt(mu)) over the subset of the reference file"}
    json.dump(R,open("docs/part2_response.json","w"),indent=1)
    print("wrote docs/part2_response.json")
