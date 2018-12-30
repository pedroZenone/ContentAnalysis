# Content Analysis

Have you ever wondered why a creative has more engagement/CTR/whatever than others? How could you querie an image and generate your own analytics?

I have implemented this project in different sources and with different KPIs, for example getting data from facebook and using Likes,love, hate, etc as engagement, your images from DCM and use the CTR, images you send in your email marketing and use the open rate as KPI, same analysis with videos and VTR, or what you want!

This is still under development, and as time goes by I will add new features.

To use it here's a quick guide:

- Vision.py: Ones you got the images or videos run this. You have to have connection to Google vision API. The fact is that this API gives me many descriptive labels of the image that can sometimes be very usefull.

- FeatureGen.py: But google features aren't enough, so I added another features, like segments of colors (ex.: dark red, light red will be just red), size of image, bright, saturation, labels understanding, face proporcion in the image, text area, etc.

- Exploratorio.py: With all this features generate visualizations in order to see if there is any difference in the distribution.


