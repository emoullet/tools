# hub
Bridge between ROS2 and arduino mega environnement. This allows you to receive information from sensors connected to the Arduino board and send commands (ex: relay, LED, servomotor, ...).

# Description

This package bridges an Arduino Mega board and ROS2 topics in order to utilise the Arduino board's inputs and outputs to connect various sensors and send commands, for example, to a relay, LEDs or servomotors.

Topic for sending an order: /hub/digital_output //
Topic for receving information : /hub/digital_input & hub/analogic_input

<img width="1434" height="803" alt="image" src="https://github.com/user-attachments/assets/b4266b68-da29-455f-847a-8bfc4554db1a" />

A message consists of at least three elements :
<type of signal> : "D" if it's a digital signal to send, "d" if it's a digital signal receved, "a" if it's a analogic signal receved.
<pin number> : just the value of the digital or analogic pin
<value to send of receved from the pin> : value egalue to 0 or 1 if it's a digital value, or the value between 0 and 255 if if's a analogic one.

You may add as much information as you wish, provided that it is of the same type and that the PIN number is indicated.

Example for sending a value with "/hub/digital_output" : data:[2,0] = send a digital value to pin "2" in state "0"

Example2 for sending a value with "/hub/digital_output" : data:[2,0,3,1] = send a digital value to pin "2" in state "0" and send a digital value to pin "3" in state "1"

Example for receved a value with "/hub/digital_input" : data:[30,1,44,1] = receve a digital value from pin "30" in state "1" and receve a digital value from pin "44" in state "1"

Example for receved a value with "/hub/analogic_input" : data[0,127,4,248] = receve an analogic value to pin "0" in state "127" and receve an analogic value from pin "4" in state "248"

You can try publishing manually to the "/hub/digital_output" topic with the following example :
ros2 topic pub -1 /hub/digital_output std_msgs/msg/Float32MultiArray "layout:
  dim: []
  data_offset: 0
data: [2.0,0.0]"
