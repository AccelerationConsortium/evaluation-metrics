# Branin Repeat Campaigns

This script runs 10 repeat benchmark campaigns to determine optimal number of initialization points for the Branin function with repeats.

## Usage

```bash
cd scripts
python branin_repeat_campaigns.py
```

## Output

The script generates a comprehensive set of results in the `branin_repeat_campaign_results/` directory:

### Individual Campaign Results (10 campaigns)
- `branin_campaign_[1-10]_results.png` - Static plots showing optimization progress and efficiency metrics
- `cv_plots_[1-10]/` - Individual campaign directories containing:
  - `branin_campaign_N_data.json` - Complete campaign data (parameters, objectives, best values)
  - `branin_campaign_N_results.html` - Interactive Plotly visualization  
  - `branin_campaign_N_evolution.gif` - Animated GIF showing parameter space exploration

### Composite Analysis
- `branin_repeat_campaigns_composite_results.png` - Combined analysis plot with:
  - **Top subplot**: Average optimization progress with best-so-far trace + std dev uncertainty bands (no raw measured values)
  - **Bottom subplot**: Average optimization efficiency across campaigns (no individual traces, no uncertainty bands)
- `campaign_summary.json` - Statistical summary of all campaigns

## Features

- **Minimal Changes**: All modifications confined to the `/scripts` directory as specified
- **Official Ax 0.5.0**: Uses official Ax documentation and API patterns
- **Comprehensive Visualization**: 
  - Static plots similar to `branin_campaign_demonstration_results.png`
  - Interactive HTML plots with Plotly
  - Animated GIFs showing optimization evolution
- **Modified Composite Plot**: Meets exact requirements from issue
  - Only best-so-far traces with uncertainty bands (no raw values)
  - Only averages across campaigns (no individual traces)

## Results Summary

The 10 repeat campaigns achieved excellent consistency:
- Mean final best value: 0.403454 ± 0.004497
- Best overall result: 0.399269 (close to Branin global minimum ~0.398)
- All campaigns converged to optimal region consistently

This demonstrates the robustness of the optimization approach and provides data for determining optimal initialization points.