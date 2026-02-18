# hub
Bridge between ROS2 and arduino mega environnement. This allows you to receive information from sensors connected to the Arduino board and send commands (ex: relay, LED, servomotor, ...).

# Description

This package bridges an Arduino Mega board and ROS2 topics in order to utilise the Arduino board's inputs and outputs to connect various sensors and send commands, for example, to a relay, LEDs or servomotors.

Topic for sending an order: /hub/digital_output //
Topic for receving information : /hub/digital_input & hub/analogic_input

<img width="1434" height="803" alt="image" src="https://github.com/user-attachments/assets/b4266b68-da29-455f-847a-8bfc4554db1a" />

A message consists of at least three elements.
<type of signal>
<pin number>
<value to send of receved from the pin>
