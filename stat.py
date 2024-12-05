import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

# Data for Population 1 and Population X (Example: Population 2)
rastrigin_NoAdap = np.array([9721,  8290,  9210, 9546, 10341, 10475, 10246,  9506,  9302, 11728])
rastrigin_OneFif = np.array([9922, 10848, 10555, 9700,  9693, 10532,  9781, 10148, 10236, 8962])
rastrigin_AvgDis = np.array([9397,  8286,  9660, 9789,  9834,  8905, 10759,  9930, 10378, 9408])
rastrigin_CosCom = np.array([9666, 8595, 8672, 11271,  9883, 8838,  10073, 10499, 9747, 9378])
rastrigin_CosCtl = np.array([10191, 9802, 9081, 10314,  9943, 8920,  10738, 10146, 10079, 11521])
rastrigin_CosAnn = np.array([8981, 9933, 9873, 9793,  10158, 9893,  9783, 9081, 9719, 9103])
rastrigin_AvgFit = np.array([10425, 10116, 10783, 9424,  9755, 8866,  10371, 8331, 9573, 9118])
rastrigin_Mating = np.array([9277, 9919, 10192, 10509,  8696, 9907,  10556, 8670, 12048, 9745])


# Perform Mann-Whitney U Test
stat_list = []
p_value_list = []
stat, p_value = mannwhitneyu(rastrigin_NoAdap, rastrigin_OneFif, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rastrigin_NoAdap, rastrigin_AvgDis, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rastrigin_NoAdap, rastrigin_CosCom, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rastrigin_NoAdap, rastrigin_CosCtl, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rastrigin_NoAdap, rastrigin_CosAnn, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rastrigin_NoAdap, rastrigin_AvgFit, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rastrigin_NoAdap, rastrigin_Mating, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)

# Print the result

for i in range(len(stat_list)):
    stat = stat_list[i]
    p_value = p_value_list[i]
    print(f"Mann-Whitney U statistic: {stat}")
    print(f"P-value: {p_value}")

# Interpret the result
    if p_value < 0.05:
        print("Reject the null hypothesis: The distributions are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The distributions are not significantly different.")


print('\n\n')
# Perform T test
stat_list = []
p_value_list = []
stat, p_value = ttest_ind(rastrigin_NoAdap, rastrigin_OneFif, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rastrigin_NoAdap, rastrigin_AvgDis, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rastrigin_NoAdap, rastrigin_CosCom, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rastrigin_NoAdap, rastrigin_CosCtl, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rastrigin_NoAdap, rastrigin_CosAnn, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rastrigin_NoAdap, rastrigin_AvgFit, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rastrigin_NoAdap, rastrigin_Mating, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)

# Print the result

for i in range(len(stat_list)):
    stat = stat_list[i]
    p_value = p_value_list[i]
    # Print the result
    print(f"T-statistic: {stat}")
    print(f"P-value: {p_value}")

    # Interpret the result
    if p_value < 0.05:
        print("Reject the null hypothesis: The means are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different.")


print("\n\n")


"""
Rosenbrock
"""
# Data for Population 1 and Population X (Example: Population 2)
rosenbrock_NoAdap = np.array([913, 279374, 259969, 196898, 378744, 754527, 152661, 430638, 277491, 526311])
rosenbrock_OneFif = np.array([460015, 176672, 445041, 825379, 447618, 10207, 2209, 362239, 198372, 862])
rosenbrock_AvgDis = np.array([266114, 2402, 417133, 301924, 636573, 82745, 349562, 428665, 970, 413759])
rosenbrock_CosCom = np.array([16507, 547218, 863, 517388, 384233, 2005, 566900, 344862, 1205, 529673])
rosenbrock_CosCtl = np.array([551260, 210012, 38416, 431624, 1257, 270898, 293283, 555294, 307306, 411770])
rosenbrock_CosAnn = np.array([556565, 1093, 128978, 688807, 445214, 1698, 1944, 1023, 553945, 381846])
rosenbrock_AvgFit = np.array([2976, 698353, 370128, 620860, 556153, 732258, 470881, 582945, 286656, 300995])
rosenbrock_Mating = np.array([646354, 440354, 658599, 375288, 383198, 224757, 644895, 696804, 602293, 62720])


# Perform Mann-Whitney U Test
stat_list = []
p_value_list = []
stat, p_value = mannwhitneyu(rosenbrock_NoAdap, rosenbrock_OneFif, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rosenbrock_NoAdap, rosenbrock_AvgDis, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rosenbrock_NoAdap, rosenbrock_CosCom, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rosenbrock_NoAdap, rosenbrock_CosCtl, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rosenbrock_NoAdap, rosenbrock_CosAnn, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rosenbrock_NoAdap, rosenbrock_AvgFit, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(rosenbrock_NoAdap, rosenbrock_Mating, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)

# Print the result

for i in range(len(stat_list)):
    stat = stat_list[i]
    p_value = p_value_list[i]
    print(f"Mann-Whitney U statistic: {stat}")
    print(f"P-value: {p_value}")

# Interpret the result
    if p_value < 0.05:
        print("Reject the null hypothesis: The distributions are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The distributions are not significantly different.")


print('\n\n')
# Perform T test
stat_list = []
p_value_list = []
stat, p_value = ttest_ind(rosenbrock_NoAdap, rosenbrock_OneFif, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rosenbrock_NoAdap, rosenbrock_AvgDis, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rosenbrock_NoAdap, rosenbrock_CosCom, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rosenbrock_NoAdap, rosenbrock_CosCtl, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rosenbrock_NoAdap, rosenbrock_CosAnn, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rosenbrock_NoAdap, rosenbrock_AvgFit, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(rosenbrock_NoAdap, rosenbrock_Mating, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)

# Print the result

for i in range(len(stat_list)):
    stat = stat_list[i]
    p_value = p_value_list[i]
    # Print the result
    print(f"T-statistic: {stat}")
    print(f"P-value: {p_value}")

    # Interpret the result
    if p_value < 0.05:
        print("Reject the null hypothesis: The means are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different.")


print("\n\n")


"""
Sphere
"""
# Data for Population 1 and Population X (Example: Population 2)
sphere_NoAdap = np.array([1750, 1628, 1278, 1524, 660, 1678, 1597, 1667, 1686, 1090])
sphere_OneFif = np.array([1259, 1640, 1528, 1888, 1178, 1749, 1995, 1635, 1069, 1987])
sphere_AvgDis = np.array([2138, 1585, 1238, 1770, 1802, 1472, 1213, 1368, 1735, 1233])
sphere_CosCom = np.array([1488, 1488, 1524, 1418, 1748, 1548, 1187, 1203, 1765, 1532])
sphere_CosCtl = np.array([1645, 1467, 1687, 1707, 1220, 1334, 932, 1665, 1347, 1626])
sphere_CosAnn = np.array([1772, 1711, 1063, 1974, 1672, 1624, 1202, 1583, 1700 ,1562])
sphere_AvgFit = np.array([864, 1614, 1855, 1570, 1410, 1302, 1956, 1325, 1481, 2062])
sphere_Mating = np.array([2283, 1830, 2122, 2239, 1968, 2031, 1749, 1726, 2484, 2036])


# Perform Mann-Whitney U Test
stat_list = []
p_value_list = []
stat, p_value = mannwhitneyu(sphere_NoAdap, sphere_OneFif, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(sphere_NoAdap, sphere_AvgDis, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(sphere_NoAdap, sphere_CosCom, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(sphere_NoAdap, sphere_CosCtl, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(sphere_NoAdap, sphere_CosAnn, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(sphere_NoAdap, sphere_AvgFit, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = mannwhitneyu(sphere_NoAdap, sphere_Mating, alternative='two-sided')
stat_list.append(stat)
p_value_list.append(p_value)

# Print the result

for i in range(len(stat_list)):
    stat = stat_list[i]
    p_value = p_value_list[i]
    print(f"Mann-Whitney U statistic: {stat}")
    print(f"P-value: {p_value}")

# Interpret the result
    if p_value < 0.05:
        print("Reject the null hypothesis: The distributions are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The distributions are not significantly different.")


print('\n\n')
# Perform T test
stat_list = []
p_value_list = []
stat, p_value = ttest_ind(sphere_NoAdap, sphere_OneFif, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(sphere_NoAdap, sphere_AvgDis, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(sphere_NoAdap, sphere_CosCom, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(sphere_NoAdap, sphere_CosCtl, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(sphere_NoAdap, sphere_CosAnn, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(sphere_NoAdap, sphere_AvgFit, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)
stat, p_value = ttest_ind(sphere_NoAdap, sphere_Mating, alternative='two-sided', equal_var=False)
stat_list.append(stat)
p_value_list.append(p_value)

# Print the result

for i in range(len(stat_list)):
    stat = stat_list[i]
    p_value = p_value_list[i]
    # Print the result
    print(f"T-statistic: {stat}")
    print(f"P-value: {p_value}")

    # Interpret the result
    if p_value < 0.05:
        print("Reject the null hypothesis: The means are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different.")


