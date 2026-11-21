"""
KARMA-X Interactive Demo - Streamlit App
Demonstrates temporal KG reasoning for multiple downstream tasks

Run with: streamlit run karma_x_streamlit_demo.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px


def get_rng(*parts):
    """Deterministic RNG seeded from the query inputs, so the same query
    always reproduces the same 'prediction', but different queries produce
    genuinely different (still plausible) outputs -- unlike a fixed mock
    table, this responds to what the user actually typed."""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.md5(text.encode()).hexdigest()
    seed = int(digest[:8], 16)
    return np.random.RandomState(seed)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="KARMA-X: Inductive Temporal KG Reasoning",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTabs [data-baseurlpath] button { font-size: 16px; font-weight: bold; }
    h1 { color: #0052CC; }
    h2 { color: #0052CC; margin-top: 2rem; }
    .metric-box {
        background: linear-gradient(135deg, #0052CC 0%, #0066FF 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: MODEL & DATASET SELECTION
# ============================================================================
with st.sidebar:
    st.header("Configuration")

    # Dataset selection
    dataset = st.selectbox(
        "Select Dataset",
        ["ICEWS14 (Diplomacy)", "ICEWS18 (Diplomacy)", "WIKI-clean (General)",
         "YAGO11k (Entity Linking)", "YAGO15k (Entity Linking)", "WIKIDATA12k (Multi-domain)"],
        index=0
    )

    # Model variant
    model_variant = st.selectbox(
        "Model Variant",
        ["KARMA-ER"],
        index=0
    )

    st.markdown("---")
    st.subheader("Model Info")
    st.metric("Anchor Entities", "128")
    st.metric("Temporal Heads", "4")
    st.metric("Hidden Dim", "256")

    st.markdown("---")
    st.info("""
    **KARMA-X** uses anchor-attention encoding to reason about
    **unseen entities** without entity embeddings. Fully inductive.
    """)

# ============================================================================
# TITLE & INTRO
# ============================================================================
st.title("KARMA-X Demo: Anchor-Attention Reasoning for Cold-Start Entities")

st.markdown("""
KARMA-X solves the **cold-start problem** in temporal knowledge graphs:
- **Problem:** Traditional models can't reason about unseen entities
- **Solution:** Anchor-attention encoding using reference entities as intermediaries
- **Impact:** Enables immediate predictions for new users, items, genes, etc.

Select a downstream task below to see KARMA-X in action.
""")

st.markdown("---")

# ============================================================================
# TABS: DIFFERENT USE CASES
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Link Prediction",
    "Recommendation (prototype)",
    "Question Answering (prototype)",
    "Biomedical (prototype)",
    "Fact Verification (prototype)",
    "Benchmarks"
])

# ============================================================================
# TAB 1: LINK PREDICTION (Main use case)
# ============================================================================
with tab1:
    st.header("Temporal Link Prediction")
    st.warning("**Interface prototype.** Outputs in this tab are simulated (deterministic pseudo-random values seeded by your input) and do not come from a trained model or a domain dataset. The tab illustrates how anchor-based evidence could be presented; see the Benchmarks tab for model results.")
    st.markdown("**Predict relationships for unseen entities in temporal KGs**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Query Setup")

        # Entity input
        entity_input = st.text_input(
            "Enter unseen entity ID:",
            value="Entity_4782",
            help="This entity was NOT in the training set"
        )

        # Relation selection
        relation = st.selectbox(
            "Predict which relation?",
            ["Any relation (?)", "Interacts with", "Condemns", "Supports", "Negotiates with"],
            index=0
        )

        # Timestamp
        timestamp = st.slider(
            "Timestamp:",
            min_value=2014,
            max_value=2018,
            value=2014,
            step=1
        )

        # Query button
        run_query = st.button("Generate Predictions", use_container_width=True)

    # Seed everything for this query off the actual inputs, so different
    # entities / relations / timestamps / datasets produce different (but
    # reproducible) results instead of a fixed mock table.
    _rng_preview = get_rng(entity_input, relation, timestamp, dataset)
    _anchor_ids = _rng_preview.choice(range(1, 400), size=3, replace=False)
    _anchor_rel = sorted(_rng_preview.uniform(0.15, 0.5, size=3), reverse=True)
    _anchor_rel = [r / sum(_anchor_rel) for r in _anchor_rel]  # normalize to ~sum 1

    with col2:
        st.subheader("Entity Information")
        st.info(f"""
        **Entity ID:** {entity_input}

        **Training Status:** UNSEEN (not in training data)

        **Encoding Method:** Anchor-Attention

        **Anchor Entities Selected:**
        - Anchor_{_anchor_ids[0]:03d} (relevance: {_anchor_rel[0]:.2f})
        - Anchor_{_anchor_ids[1]:03d} (relevance: {_anchor_rel[1]:.2f})
        - Anchor_{_anchor_ids[2]:03d} (relevance: {_anchor_rel[2]:.2f})
        """)

    if run_query:
        st.markdown("---")
        st.subheader("Prediction Results")

        # Predictions are generated deterministically from entity_input,
        # relation, timestamp, and dataset -- change any of them and the
        # results below change too.
        rng = get_rng(entity_input, relation, timestamp, dataset)
        relation_pool = ["Interacts with", "Condemns", "Supports", "Negotiates with",
                          "Meets with", "Cooperates with", "Criticizes", "Praises"]
        target_pool = ["USA", "Russia", "UK", "France", "Germany", "China",
                        "Japan", "India", "Brazil", "Canada"]

        chosen_relations = list(rng.choice(relation_pool, size=5, replace=False))
        chosen_targets = list(rng.choice(target_pool, size=5, replace=False))
        confidences = sorted(rng.uniform(0.60, 0.95, size=5), reverse=True)
        attn_weights = sorted(rng.uniform(0.10, 0.48, size=5), reverse=True)

        predictions_data = {
            "Rank": [1, 2, 3, 4, 5],
            "Predicted Relation": chosen_relations,
            "Target Entity": chosen_targets,
            "Confidence": [round(c, 2) for c in confidences],
            "Attention Weight": [round(a, 2) for a in attn_weights]
        }

        pred_df = pd.DataFrame(predictions_data)

        # Display table
        col_table, col_chart = st.columns([1.2, 1])

        with col_table:
            st.dataframe(
                pred_df.style.format({"Confidence": "{:.2f}", "Attention Weight": "{:.2f}"}),
                use_container_width=True,
                hide_index=True
            )

        with col_chart:
            # Confidence chart
            fig = px.bar(
                pred_df,
                x="Predicted Relation",
                y="Confidence",
                color="Confidence",
                color_continuous_scale="Blues",
                title="Prediction Confidence Scores"
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Anchor attention visualization
        st.subheader("Anchor Attention Mechanism")

        relevance_scores = sorted(rng.uniform(0.55, 0.92, size=3), reverse=True)
        anchor_data = {
            "Anchor Entity": [
                f"Anchor_{_anchor_ids[0]:03d} ({chosen_targets[0]})",
                f"Anchor_{_anchor_ids[1]:03d} ({chosen_targets[1]})",
                f"Anchor_{_anchor_ids[2]:03d} ({chosen_targets[2]})",
            ],
            "Attention Weight": [round(_anchor_rel[0], 2), round(_anchor_rel[1], 2), round(_anchor_rel[2], 2)],
            "Relevance Score": [round(r, 2) for r in relevance_scores]
        }
        anchor_df = pd.DataFrame(anchor_data)

        fig_anchor = go.Figure(data=[
            go.Bar(x=anchor_df["Anchor Entity"], y=anchor_df["Attention Weight"],
                   marker_color='#0052CC', name="Attention Weight"),
            go.Bar(x=anchor_df["Anchor Entity"], y=anchor_df["Relevance Score"],
                   marker_color='#FFB81C', name="Relevance Score")
        ])
        fig_anchor.update_layout(
            title="How KARMA-X Reasons About the Unseen Entity",
            barmode='group',
            height=300
        )
        st.plotly_chart(fig_anchor, use_container_width=True)

        st.success("""
        **Result:** KARMA-X successfully predicted relations for unseen entity
        using anchor-based reasoning. No entity embeddings needed!
        """)


# ============================================================================
# TAB 2: RECOMMENDATION SYSTEM
# ============================================================================
with tab2:
    st.header("Cold-Start Recommendation System")
    st.warning("**Interface prototype.** Outputs in this tab are simulated (deterministic pseudo-random values seeded by your input) and do not come from a trained model or a domain dataset. The tab illustrates how anchor-based evidence could be presented; see the Benchmarks tab for model results.")
    st.markdown("**Recommend items to new users without historical data**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("New User Profile")

        new_user = st.text_input("New user ID:", value="User_987654")

        user_profile = st.multiselect(
            "User interests (select anchors):",
            ["Action Movies", "Sci-Fi", "Drama", "Comedy", "Documentaries"],
            default=["Action Movies", "Sci-Fi"]
        )

        st.info(f"**Status:** New user (0 historical interactions)")
        st.info(f"**Anchor Users:** 5 similar users identified")

        recommend_btn = st.button("Get Recommendations", use_container_width=True)

    with col2:
        st.subheader("Anchor User Analysis")
        _rng_users = get_rng(new_user, tuple(sorted(user_profile)))
        anchor_names = ["Power User_42", "Cinephile_103", "Action Fan_217",
                         "Sci-Fi Geek_445", "Drama Lover_562", "Binge Watcher_318",
                         "Critic_775", "Casual Viewer_901"]
        chosen_anchors = list(_rng_users.choice(anchor_names, size=5, replace=False))
        anchor_users = pd.DataFrame({
            "Anchor User": chosen_anchors,
            "Similarity": [round(s, 2) for s in sorted(_rng_users.uniform(0.72, 0.94, size=5), reverse=True)],
            "Avg Rating": [round(r, 1) for r in _rng_users.uniform(3.8, 4.8, size=5)]
        })
        st.dataframe(anchor_users, use_container_width=True, hide_index=True)

    if recommend_btn:
        st.markdown("---")
        st.subheader("Recommended Items")

        # Movie pool is filtered toward the user's selected interests when
        # possible, then results are seeded from the user ID + interests so
        # a different profile yields different recommendations.
        movie_pool = {
            "Action Movies": [("Mad Max: Fury Road", "Action"), ("John Wick", "Action")],
            "Sci-Fi": [("Inception", "Sci-Fi/Action"), ("Interstellar", "Sci-Fi/Drama"),
                       ("Blade Runner 2049", "Sci-Fi"), ("Dune", "Sci-Fi/Action"),
                       ("The Matrix", "Sci-Fi/Action")],
            "Drama": [("The Shawshank Redemption", "Drama"), ("Parasite", "Drama/Thriller")],
            "Comedy": [("The Grand Budapest Hotel", "Comedy"), ("Knives Out", "Comedy/Mystery")],
            "Documentaries": [("Free Solo", "Documentary"), ("My Octopus Teacher", "Documentary")],
        }
        candidates = []
        for interest in user_profile:
            candidates.extend(movie_pool.get(interest, []))
        if not candidates:
            candidates = [m for lst in movie_pool.values() for m in lst]
        candidates = list(dict.fromkeys(candidates))  # dedupe, keep order

        rng_rec = get_rng(new_user, tuple(sorted(user_profile)), "recs")
        n_picks = min(5, len(candidates))
        idx = rng_rec.choice(len(candidates), size=n_picks, replace=False)
        picked = [candidates[i] for i in idx]
        ratings = sorted(rng_rec.uniform(4.0, 4.95, size=n_picks), reverse=True)
        confidences = sorted(rng_rec.uniform(0.75, 0.96, size=n_picks), reverse=True)

        recommendations = pd.DataFrame({
            "Rank": list(range(1, n_picks + 1)),
            "Movie Title": [p[0] for p in picked],
            "Genre": [p[1] for p in picked],
            "Pred. Rating": [round(r, 1) for r in ratings],
            "Confidence": [round(c, 2) for c in confidences]
        })

        st.dataframe(
            recommendations.style.format({"Pred. Rating": "{:.1f}", "Confidence": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        fig_rec = px.bar(
            recommendations,
            x="Movie Title",
            y="Confidence",
            color="Confidence",
            color_continuous_scale="Viridis",
            title="Recommendation Confidence Scores"
        )
        fig_rec.update_layout(height=300)
        st.plotly_chart(fig_rec, use_container_width=True)

        st.success(f"""
        **Success!** Generated {len(recommendations)} personalized recommendations
        for new user {new_user} using anchor-based collaborative filtering.
        """)


# ============================================================================
# TAB 3: QUESTION ANSWERING
# ============================================================================
with tab3:
    st.header("Question Answering on Emerging Entities")
    st.warning("**Interface prototype.** Outputs in this tab are simulated (deterministic pseudo-random values seeded by your input) and do not come from a trained model or a domain dataset. The tab illustrates how anchor-based evidence could be presented; see the Benchmarks tab for model results.")
    st.markdown("**Answer questions about entities not in training data**")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Ask a Question")

        question = st.text_area(
            "Question about emerging entity:",
            value="What countries does the new diplomatic mission [DM_5847] interact with?",
            height=100
        )

        qa_btn = st.button("Find Answer", use_container_width=True)

    with col2:
        st.subheader("Entity Context")
        st.info("""
        **Entity:** New Diplomatic Mission

        **Training Status:** Unseen

        **Known Info:**
        - Founded: 2024-03-15
        - Location: Middle East
        - Type: Regional Hub
        """)

    if qa_btn:
        st.markdown("---")
        st.subheader("Answer")

        # Seeded from the actual question text, so a different question
        # produces a different set of countries/interactions/confidences.
        rng_qa = get_rng(question)
        country_pool = ["USA", "France", "Germany", "UK", "Japan", "Canada",
                         "Brazil", "India", "Italy", "South Korea"]
        interaction_pool = ["Diplomatic negotiation", "Trade agreement", "Cultural exchange",
                             "Intelligence sharing", "Tourism partnership", "Joint task force",
                             "Aid coordination"]
        anchor_evidence_pool = ["Embassy", "Consulate", "Trade Office", "Liaison Office",
                                 "Cultural Center", "Regional Bureau"]

        chosen_countries = list(rng_qa.choice(country_pool, size=5, replace=False))
        chosen_interactions = list(rng_qa.choice(interaction_pool, size=5, replace=False))
        confidences = sorted(rng_qa.uniform(0.65, 0.92, size=5), reverse=True)
        anchor_ids = rng_qa.choice(range(1, 400), size=5, replace=False)
        anchor_kinds = rng_qa.choice(anchor_evidence_pool, size=5, replace=True)
        evidence = [f"Anchor_{aid:03d} ({c} {kind})" for aid, c, kind in
                    zip(anchor_ids, chosen_countries, anchor_kinds)]

        answer_df = pd.DataFrame({
            "Rank": [1, 2, 3, 4, 5],
            "Country": chosen_countries,
            "Interaction Type": chosen_interactions,
            "Confidence": [round(c, 2) for c in confidences],
            "Evidence": evidence
        })

        st.dataframe(
            answer_df.style.format({"Confidence": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        summary = ", ".join(
            f"{c} ({conf*100:.0f}%)" for c, conf in zip(chosen_countries, confidences)
        )
        st.info(f"""
        **Answer:** Based on the question, KARMA-X identifies {len(chosen_countries)}
        likely interactions: {summary}.

        Reasoning: KARMA-X encoded the unseen entity via anchor entities
        (established embassies/consulates) and predicted relations based on
        their historical interactions.
        """)


# ============================================================================
# TAB 4: BIOMEDICAL DISCOVERY
# ============================================================================
with tab4:
    st.header("Biomedical Knowledge Graph: Gene-Protein Interactions")
    st.warning("**Interface prototype.** Outputs in this tab are simulated (deterministic pseudo-random values seeded by your input) and do not come from a trained model or a domain dataset. The tab illustrates how anchor-based evidence could be presented; see the Benchmarks tab for model results.")
    st.markdown("**Predict protein interactions for newly discovered genes**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("New Gene Discovery")

        gene_id = st.text_input("Gene ID:", value="GENE_XYZ_2024")

        gene_info = st.text_area(
            "Known gene properties:",
            value="Chromosome: 12\nExpression: High in leukocytes\nFunction: Immunosuppression\nRelated: Similar to anchor genes STAT1, STAT3",
            height=120
        )

        gene_btn = st.button("Predict Interactions", use_container_width=True)

    with col2:
        st.subheader("Anchor Genes (Reference)")
        anchor_genes = pd.DataFrame({
            "Anchor Gene": ["STAT1", "STAT3", "FOXP3", "IL10"],
            "Similarity": [0.92, 0.88, 0.81, 0.76],
            "Known Proteins": ["p-STAT1", "p-STAT3", "FOXP3", "IL-10R"]
        })
        st.dataframe(anchor_genes, use_container_width=True, hide_index=True)

    if gene_btn:
        st.markdown("---")
        st.subheader("Predicted Protein Interactions")

        # Seeded from gene_id + the stated gene properties, so a different
        # gene / description yields different predicted targets.
        rng_gene = get_rng(gene_id, gene_info)
        protein_pool = ["SOCS3", "IL-6 Receptor", "TNF-alpha", "TGF-beta", "IFN-gamma",
                         "JAK2", "IL-10R", "NF-kB", "mTOR", "PD-L1"]
        interaction_types = ["Binding", "Inhibition", "Signal transduction",
                              "Co-activation", "Immune suppression", "Phosphorylation"]
        relevance_levels = ["High", "Medium", "Low"]

        chosen_proteins = list(rng_gene.choice(protein_pool, size=5, replace=False))
        chosen_interactions = list(rng_gene.choice(interaction_types, size=5, replace=True))
        confidences = sorted(rng_gene.uniform(0.68, 0.96, size=5), reverse=True)
        relevance = [relevance_levels[0] if c > 0.85 else relevance_levels[1] if c > 0.75 else relevance_levels[2]
                     for c in confidences]

        interactions = pd.DataFrame({
            "Rank": [1, 2, 3, 4, 5],
            "Protein Target": chosen_proteins,
            "Interaction Type": chosen_interactions,
            "Confidence": [round(c, 2) for c in confidences],
            "Relevance": relevance
        })

        st.dataframe(
            interactions.style.format({"Confidence": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.success(f"""
        **Biomedical Discovery:** Predicted {len(interactions)} likely protein
        interactions for newly discovered {gene_id}. These predictions can guide
        experimental validation.
        """)

        st.info("""
        **Next Steps:**
        - Validate top predictions via wet-lab experiments
        - Test therapeutic potential of predicted interactions
        - Update biomedical KG with validated results
        """)


# ============================================================================
# TAB 5: FACT VERIFICATION
# ============================================================================
with tab5:
    st.header("Fact Verification for Emerging Claims")
    st.warning("**Interface prototype.** Outputs in this tab are simulated (deterministic pseudo-random values seeded by your input) and do not come from a trained model or a domain dataset. The tab illustrates how anchor-based evidence could be presented; see the Benchmarks tab for model results.")
    st.markdown("**Verify news claims about new actors in temporal KGs**")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Claim to Verify")

        claim = st.text_area(
            "Enter a claim:",
            value="The new technology startup TechStartup_2024 signed a partnership with Google and Microsoft in Q2 2024.",
            height=120
        )

        verify_btn = st.button("Verify Claim", use_container_width=True)

    with col2:
        st.subheader("Temporal Context")
        st.info("""
        **Entity:** TechStartup_2024

        **Status:** New entity (founded 2024)

        **Verification Window:** Q2 2024

        **Question:** Is the partnership claim likely?
        """)

    if verify_btn:
        st.markdown("---")
        st.subheader("Verification Result")

        # Seeded from the claim text -- a different claim yields a
        # different overall confidence and component breakdown.
        rng_fv = get_rng(claim)
        components = ["Partnership likelihood", "Anchor-1 historical pattern",
                      "Anchor-2 historical pattern", "Timing coherence", "Sector relevance"]
        scores = rng_fv.uniform(0.55, 0.96, size=5)
        overall = float(np.mean(scores))
        results_icons = ["Pass" if s >= 0.7 else "Caution" for s in scores]

        col_result, col_chart = st.columns([1, 1])

        with col_result:
            st.metric("Claim Confidence", f"{overall:.2f}",
                       f"{overall - 0.72:+.2f}", delta_color="off")

            verification_breakdown = pd.DataFrame({
                "Component": components,
                "Score": [round(s, 2) for s in scores],
                "Result": results_icons
            })
            st.dataframe(verification_breakdown, use_container_width=True, hide_index=True)

        with col_chart:
            fig_verify = go.Figure(data=[
                go.Scatterpolar(
                    r=[round(s, 2) for s in scores],
                    theta=["Partnership", "Anchor-1 Pattern", "Anchor-2 Pattern", "Timing", "Relevance"],
                    fill='toself',
                    name='Verification Scores'
                )
            ])
            fig_verify.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_verify, use_container_width=True)

        verdict = "HIGHLY LIKELY" if overall >= 0.8 else "PLAUSIBLE" if overall >= 0.6 else "UNCERTAIN"
        st.success(f"""
        **Verification Result:** {verdict} ({overall*100:.0f}% confidence)

        **Reasoning:**
        - Anchor entities relevant to the claim have established historical patterns
        - Timing and sector context were checked against those patterns
        - Score reflects agreement between the claim and anchor-based evidence
        """)

        st.warning("**Note:** High confidence indicates claim is *plausible* based on KG patterns, not definitively true.")


# ============================================================================
# TAB 6: BENCHMARKS
# ============================================================================
with tab6:
    st.header("Benchmark Results")
    import json, os
    _f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "summary.json")
    if not os.path.exists(_f):
        st.info("results/summary.json not found. Generate it on the server with export_demo_results.py.")
    else:
        R = json.load(open(_f))
        st.caption(R.get("note", ""))
        def show(title, key, cols=None, help_text=None):
            rows = R.get(key, [])
            st.subheader(title)
            if help_text:
                st.markdown(help_text)
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)
            else:
                st.write("Not available yet.")
        show("(a) Link prediction: unseen-entity MRR (ACIS)", "link_prediction",
             help_text="KARMA-ER: mean (std) over seeds. KARMA-X: best of 8 earlier variants under ACIS.")
        show("Unseen-entity MRR by number of facts the entity had before the query", "buckets")
        show("(b) Fact verification: AUROC", "fact_verification",
             help_text="`_new` = claims whose object is a new entity; `negatives` = how corrupted claims were drawn.")
        show("(c) Recommendation: MovieLens-1M, new user -> liked item", "recommendation")
        show("(d) Biomedical: STRING, held-out proteins (3 known interactions)", "biomedical")
        with st.expander("Protocol audit: earlier KARMA-X variants under TOIS vs ACIS"):
            show("All variants, both history settings", "protocol_audit")
            show("Random-data control (chance MRR about 0.024)", "random_control")
        with st.expander("Ablations of KARMA-ER"):
            show("Component removed", "ablations")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns([1, 1, 1])

with footer_col1:
    st.subheader("Paper")
    st.write("AAAI-27 Demonstration Program")

with footer_col2:
    st.subheader("Code")
    st.write("[GitHub: github.com/amangelbhullar/karma-x](https://github.com/amangelbhullar/karma-x)")

with footer_col3:
    st.subheader("Authors")
    st.write("Amangel Bhullar, Ziad Kobti | University of Windsor")

st.markdown("""
---
**KARMA-X: Fully Inductive Temporal Knowledge Graph Reasoning**

Anchor-attention encoding enables cold-start entity prediction across multiple downstream tasks:
link prediction, recommendation systems, question answering, biomedical discovery, and fact verification.
""")
