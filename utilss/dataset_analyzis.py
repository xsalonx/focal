import numpy as np


def clean_from_nans_and_prepare_report(responses, inf_energy=1e10):
    responses = responses.copy()
    
    responses_no = responses.shape[0]
    total_size = responses.size
    responses_nans_mask = np.isnan(responses)
    total_nans = responses_nans_mask.sum()
    responses_with_nans_mask = responses_nans_mask.sum(axis=2).sum(axis=1) > 0
    total_responses_with_nans = responses_with_nans_mask.sum()

    responses[responses_nans_mask] = 0
    negative_values_mask = responses < 0
    responses[negative_values_mask] = 0
    negative_values_no = negative_values_mask.sum()

    inf_values_mask = responses >= inf_energy
    responses[inf_values_mask] = 0

    report = \
    f"""
Total responses: {responses_no}
Responses shape: {responses.shape[1:3]}
Total size: {total_size}
NaN cells in all responses: {total_nans} - {total_nans/total_size*100:.3f}%
Responses with NaNs {total_responses_with_nans} - {total_responses_with_nans/responses_no*100:.3f}%
Negative cells in all responses: {negative_values_no} - {negative_values_no/total_size*100:.3f}%
Inf cells in all responses (inf={inf_energy:.2e}): {inf_values_mask.sum()} - {inf_values_mask.sum()/total_size*100:.3f}%
"""
    
    return responses, report 