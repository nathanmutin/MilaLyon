import numpy as np

# Default missing codes for all features
DEFAULT_MISSING = np.array([7, 9])

# Exceptions: grouped by shared missing codes
EXCEPTIONS = {
    tuple(range(0, 24)): np.array([]),
    tuple([216, 217, 218, 248, 249, 250, 251, 253, 254, 266, 267, 268, 269, 270, 271, 276, 277, 285, 286, 291, 292, 295, 296, 299, 300, 301, 302, 303, 304]): np.array([]),
    tuple([252]): np.array([99999]),
    tuple([262]): np.array([900]),
    tuple([264, 287, 288, 293, 294, 297]): np.array([99900]),
    tuple([242]): np.array([9]),
    tuple([246]): np.array([14]),
    tuple([247]): np.array([3]),
    tuple([88]): np.array([98, 77, 99]),
    tuple([60, 78, 80, 98, 119, 122, 168, 224, 225, 239, 240, 102, 106]): np.array([77, 99]),
    tuple([27, 28, 29, 79, 112, 114, 206, 207, 208, 209, 210, 211, 212, 213]): np.array([77, 88, 99]),
    tuple([33, 58, 99, 115, 118, 132, 151, 152, 153, 154, 192, 193]): np.array([7, 8, 9]),
    tuple([147, 148]): np.array([88, 98]),
    tuple([195, 197]): np.array([97, 98, 99]),
    tuple([49, 145]): np.array([98, 99]),
    tuple([59]): np.array([88, 99]),
    tuple([62, 63]): np.array([7777, 9999]),
    tuple([75, 130]): np.array([8, 77, 99]),
    tuple([77, 94, 110, 150]): np.array([777, 888, 999]),
    tuple(range(81, 87)): np.array([555, 777, 999]),
    tuple([89, 90, 92, 93]): np.array([777, 999]),
    tuple([91, 113]): np.array([88, 98, 77, 99]),
    tuple([101, 105]): np.array([777777, 999999]),
    tuple([111, 143]): np.array([555, 777, 888, 999]),
    tuple([127, 128]): np.array([7, 8]),
    tuple([129, 137, 138, 139, 140]): np.array([5, 7, 9]),
    tuple([131]): np.array([5, 7, 8, 9]),
    tuple([148, 149]): np.array([88, 98, 99]),
}
