## Settings
<div class='options-message'></div>
Customize the behavior of the app using the following options:


- #### File Dialog Type
  Test how these options work on your operating systems. Some combinations are buggy. 
  - <div class='radio' data-store='fd-type' data-val='native'>Use native filesystem file picker</div>
  - <div class='radio' data-store='fd-type' data-val='qt'>Use Qt file picker</div>


- #### File Dialog Behavior
  When opening a file browser, do you want it to open to the most recent location, or always start at the same place?
  - <div class='radio' data-store='fd-behavior' data-val='stay'>Always start from the same place</div>
  - - Source files: <span class='path' data-store='fd-stay-source'></span>  
  - - Destination folder: <span class='path' data-store='fd-stay-dest'></span>  
  - <div class='radio' data-store='fd-behavior' data-val='follow'>Remember most recent location</div>
  - - Source files: <span class='path' data-store='fd-follow-source'></span>  
  - - Destination folder: <span class='path' data-store='fd-follow-dest'></span>  
