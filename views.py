def render(con, stack, viz, red, k, mn, trace_text="",
           a_layer=0, a_head=0, t_pos=0):
    cur = nav_current(stack)
    mdl, tok = load(mn)
    E = embed_matrix(mdl)
    voc = vocab(con, tok)
    figs = {"main": empty_fig(), "detail": empty_fig(), "bars": empty_fig()}
    tex, info = "", ""

    try:
        if cur["level"] == "model":
            con.print("[dim]Rendering model overview...[/dim]")
            sidx = sample_indices(len(E))
            Esub = E[sidx]
            r = reduce(Esub, method=red)
            labs = cluster(Esub, k=k)
            smry = {}
            for c in np.unique(labs):
                if c < 0: continue
                gids = sidx[labs == c]
                ex = [voc.get(int(i), "?").strip() or "." for i in gids[:10]]
                smry[int(c)] = ex
            hover = [", ".join(smry.get(int(labs[i]), ["?"]))[:60] for i in range(len(Esub))]
            figs["main"] = fig_scatter(r["coords"], labs, hover,
                                        f"Embedding Space -- {mn} ({len(sidx)} tokens)")
            tex = tex_overview(E, len(sidx))
            pca_info = r.get("info", {})
            if pca_info:
                tex += " \\quad " + tex_pca(pca_info)
            st = manifold_stats(Esub, labs)
            lines = [f"**C{c}** ({st[c]['size']} tok): {', '.join(smry.get(c, []))}"
                     for c in sorted(smry)]
            info = "  \n".join(lines[:15])
            if viz == "parallel":
                figs["detail"] = fig_parallel(Esub, labs)

        elif cur["level"] == "cluster":
            cid = cur["id"]
            con.print(f"[dim]Rendering cluster {cid}...[/dim]")
            sidx = sample_indices(len(E))
            Esub = E[sidx]
            labs = cluster(Esub, k=k)
            mask = np.where(labs == cid)[0]
            if len(mask) == 0:
                figs["main"] = empty_fig("Empty cluster")
                return figs, tex, info
            sub = Esub[mask]
            r = reduce(sub, method=red)
            global_ids = sidx[mask]
            hover = [voc.get(int(global_ids[i]), "?") for i in range(len(sub))]
            figs["main"] = fig_scatter(r["coords"], np.zeros(len(sub), dtype=int), hover,
                                        f"Cluster {cid} -- {len(mask)} tokens")
            st = manifold_stats(Esub, labs)
            tex = tex_cluster(st, cid) if cid in st else ""
            ex = [voc.get(int(i), "?").strip() or "." for i in global_ids[:20]]
            info = f"**Cluster {cid}** -- {len(mask)} tokens  \n"
            info += f"Examples: {', '.join(ex)}"
            if viz == "parallel" and len(sub) > 2:
                figs["detail"] = fig_parallel(sub, np.zeros(len(sub), dtype=int))

        elif cur["level"] == "token":
            tid = cur["id"]
            vec = E[tid]
            con.print(f"[dim]Rendering token '{voc.get(tid, '?').strip()}'...[/dim]")
            if nrm(vec) < 1e-9:
                figs["main"] = empty_fig(f"Token {tid} has zero embedding")
                info = f"**'{voc.get(tid, '?').strip()}'** has a zero/near-zero embedding vector."
                tex = tex_token(vec)
                return figs, tex, info
            nbr_idx, nbr_sim = neighbors(E, tid, 30)
            all_ids = np.concatenate([[tid], nbr_idx])
            sub = E[all_ids]
            r = reduce(sub, method=red)
            hover = [voc.get(int(i), "?") for i in all_ids]
            labs = np.array([1] + [0] * len(nbr_idx))
            figs["main"] = fig_scatter(r["coords"], labs, hover,
                                        f"Neighbors of '{voc.get(tid, '?').strip()}'")
            figs["bars"] = fig_bars(vec) if viz in ("scatter", "bars") else fig_radar(vec)
            tex = tex_token(vec)
            od = outlier_dims(vec, E)
            tex += f" \\quad \\text{{outlier dims: }}{od[:5]}"
            nlist = [f"{voc.get(int(nbr_idx[i]), '?').strip()} ({nbr_sim[i]:.3f})"
                     for i in range(min(15, len(nbr_idx)))]
            info = f"**'{voc.get(tid, '?').strip()}'** -- neighbors:  \n" + ", ".join(nlist)

        elif cur["level"] == "trace":
            if not trace_text:
                figs["main"] = empty_fig("Enter text and click Trace")
                return figs, tex, info
            con.print(f"[dim]Tracing '{trace_text}' through layers...[/dim]")
            acts = activations(mdl, tok, trace_text)
            tids = token_ids(tok, trace_text)
            tidx = min(t_pos, len(tids) - 1)
            path = token_path(acts, tidx)

            # Find nearest word at each layer
            tw = trace_words(acts, E, voc, tidx)
            ll = [f"L{r['layer']}: {r['words'][0][0]}" for r in tw]
            figs["main"] = fig_path(path, ll)

            dp = delta_path(path)
            if len(dp) > 0:
                figs["bars"] = fig_bars(dp.mean(0), "Mean delta across layers")
            tex = tex_delta()

            tstr = [voc.get(t, "?").strip() for t in tids]
            info = f"Tracing **'{tstr[tidx]}'** (pos {tidx}) through {len(path)} layers  \n\n"
            info += "| Layer | Nearest Words | ‖h‖ |  \n|---|---|---|  \n"
            for r in tw:
                words_str = ", ".join(f"{w} ({s:.2f})" for w, s in r["words"])
                info += f"| L{r['layer']} | {words_str} | {r['norm']:.2f} |  \n"

            # Print trace table to console too
            table = Table(title=f"Token trace: '{tstr[tidx]}'")
            table.add_column("Layer", style="cyan")
            table.add_column("Nearest Word", style="green")
            table.add_column("Sim", style="yellow")
            table.add_column("‖h‖", style="dim")
            for r in tw:
                table.add_row(
                    f"L{r['layer']}",
                    r["words"][0][0],
                    f"{r['words'][0][1]:.3f}",
                    f"{r['norm']:.2f}"
                )
            con.print(table)

        elif cur["level"] == "attention":
            if not trace_text:
                figs["main"] = empty_fig("Enter text and click Attention")
                return figs, tex, info
            con.print(f"[dim]Rendering attention for '{trace_text}'...[/dim]")
            am = attn_maps(mdl, tok, trace_text)
            tids = token_ids(tok, trace_text)
            tokens = [voc.get(t, "?").strip() for t in tids]
            ly = min(a_layer, len(am) - 1)
            hd = min(a_head, am[ly].shape[0] - 1)
            figs["main"] = fig_attn(am[ly], tokens, hd, f"Layer {ly} Head {hd}")
            tex = f"\\text{{Layer }}={ly},\\; \\text{{Head }}={hd}"
            info = f"**Attention** -- Layer {ly}/{len(am)-1}, Head {hd}  \n"
            info += f"Tokens: {' -> '.join(tokens)}"

        elif cur["level"] == "compare":
            ids = cur.get("ids", [])
            if len(ids) >= 2:
                con.print(f"[dim]Comparing {len(ids)} tokens...[/dim]")
                sm = compare_tokens(E, ids)
                lb = [voc.get(int(i), "?").strip() for i in ids]
                figs["main"] = fig_heatmap(sm, lb, lb, "Token Similarity")
                sd = shared_dims(E, ids)
                figs["bars"] = fig_bars(E[ids].mean(0),
                    f"Shared profile -- top dims: {sd[:5]}")
                tex = f"\\text{{Comparing }} {len(ids)} \\text{{ tokens}}"
                info = f"**Comparing:** {', '.join(lb)}  \n"
                info += f"Shared dims: {sd[:10]}"

    except Exception as e:
        con.print_exception(show_locals=False)
        figs["main"] = empty_fig(f"Error: {str(e)[:80]}")

    return figs, tex, info

