# src/utils/rewards.py
import torch

def calculate_delta_m1(prediction, story):
    """
    Calculate Δ𝑀1 = M(Prediction, Edited Ending) - M(Prediction, Original Ending)
    """
    m_pred_edited = similarity(prediction, story['edited_ending'])
    m_pred_original = similarity(prediction, story['original_ending'])
    delta_m1 = m_pred_edited - m_pred_original
    return delta_m1

def calculate_delta_m2(prediction, story):
    """
    Calculate Δ𝑀2 = M(Prediction, Counterfactual) - M(Edited Ending, Counterfactual)
    """
    m_pred_counterfactual = similarity(prediction, story['counterfactual'])
    m_edited_counterfactual = similarity(story['edited_ending'], story['counterfactual'])
    delta_m2 = m_pred_counterfactual - m_edited_counterfactual
    return delta_m2

def similarity(text1, text2):
    """
    Simple similarity function between two texts.
    """
    set1, set2 = set(text1.split()), set(text2.split())
    return len(set1 & set2) / len(set1 | set2)
