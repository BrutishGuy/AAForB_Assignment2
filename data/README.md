# Data Description and Feature Engineering

This folder contains raw data files used in the project. Below, one can also find a description of data issues handled during data cleaning.
Additionally, we also describe feature engineering steps in this section

## Data Cleaning 

There are some issues in the data which need to be addressed before modelling. They include

- The screen_surface field contains specific values: "Matte", "Glossy", "matte", and "glossy". These are identified as 4 separate categories due to case-sensitivity, however, they must be grouped as 2 unique categories. Values "matte" and "glossy" are to be recoded in preprocessing.
- Handling of missing values. Perhaps SMOTE or some other inference technique is not out of the question here, simply because it is likely that similar laptops will have similar characteristics. Variables with missing values include the os variables, gpu variables, screen_surface, weight, cpu_details and detachable_keyboard.

## Feature Engineering

We propose the following feature engineering steps to be used in the modelling of price ranges (min - max price):

- SSD size grouping: We create bins describing SSD sizes as ranging from small, to medium to large. The ranges of these bins are decided via an EDA and histogramming of values present in the data.
- General storage size grouping: We create bins describing general storage sizes as ranging from small, to medium, to large. The ranges of these bins are decided as for the SSD case above.
- HD description: This variable is derived from the pixel_y and pixel_x fields and is a grouping variable designed to express the quality of the screen of the laptop. The values here are SD, HD, Full HD and 4K HD/Ultra HD.
- Weight/Portability grouping: We use weight as a metric together with screen size to designate whether a laptop is light and small, medium, or heavy.
- Notebook/Ultrabook: We use string matching to obtain whether a given laptop is a notebook, ultrabook, gaming laptop, or other.
- Brand description: We use string matching to obtain whether a given laptop is a top brand such as HP, Asus, Dell, Razor, Alienware, etc. or other lower tier brands. Pricing variation may be brand specific, which is why we group individual laptop names, which may not be useful on their own, into a higher level brand variable.
- CPU generation: Higher level description of the CPU generation
- CPU processor frequency: The processing frequency of the CPU.
- No. of Cores: The number of cores of the CPU in question.
- Special CPU: Does the CPU have any special characteristics? Is the CPU for example an i7 extreme edition?
- GPU generation: Is the GPU current or older? 
- GPU Memory: How much memory, in GB, does the GPU have on the card itself?

Motivations for having so many categorical grouping variables is due the small dataset size. Given the large number of features and the presence of various numerical variables, we attempt to reduce the complexity of the solution space by introducing simpler features which attempt to express the same information to our modelling approach.



