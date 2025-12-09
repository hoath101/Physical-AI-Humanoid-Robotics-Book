# LLM Planning for Robotics

Large Language Models (LLMs) play a crucial role in robotics by enabling natural language understanding, task planning, and high-level decision making. This section covers how to integrate LLMs with humanoid robots for intelligent task planning and execution.

## Introduction to LLMs in Robotics

LLMs bring several capabilities to robotics:

- **Natural Language Understanding**: Interpret human commands in natural language
- **Task Planning**: Decompose high-level goals into executable robot actions
- **Reasoning**: Apply logical reasoning to handle novel situations
- **Knowledge Integration**: Access vast knowledge bases for decision making
- **Human-Robot Interaction**: Enable natural communication with humans

### Key Capabilities for Robotics

1. **Command Interpretation**: Convert natural language commands to robot actions
2. **Task Decomposition**: Break down complex tasks into atomic robot operations
3. **Context Awareness**: Understand the environment and robot capabilities
4. **Error Handling**: Generate recovery strategies when tasks fail
5. **Learning**: Adapt behavior based on past experiences

## LLM Integration Approaches

### 1. OpenAI GPT Models

The most straightforward approach uses OpenAI's GPT models:

```python
import openai
import json
from typing import Dict, List, Optional

class RobotLLMInterface:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        openai.api_key = api_key
        self.model = model
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for robot task planning"""
        return """
        You are a robot task planner that converts natural language commands into robot actions.
        Your role is to:
        1. Understand the human command in natural language
        2. Decompose the command into specific robot actions
        3. Consider the robot's capabilities and environment
        4. Generate a sequence of executable actions
        5. Include error handling and validation

        Robot capabilities include:
        - Navigation: Move to specific locations
        - Manipulation: Pick up, place, grasp objects
        - Perception: Detect and recognize objects
        - Communication: Respond to human commands

        Respond in JSON format with the following structure:
        {
          "task_breakdown": [
            {
              "step": 1,
              "action": "action_type",
              "parameters": {"param1": "value1", ...},
              "description": "Human-readable description",
              "validation": "How to verify success"
            }
          ],
          "potential_issues": ["issue1", "issue2"],
          "success_criteria": "How to know the task is complete"
        }
        """

    def plan_task(self, command: str, robot_state: Dict, environment: Dict) -> Dict:
        """Plan a task based on natural language command"""
        user_prompt = f"""
        Command: {command}

        Robot State: {json.dumps(robot_state, indent=2)}
        Environment: {json.dumps(environment, indent=2)}

        Generate a detailed plan to execute this command.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Extract and parse JSON response
            response_text = response.choices[0].message.content

            # Find JSON in response (in case of additional text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_content = response_text[json_start:json_end]
                plan = json.loads(json_content)
                return plan
            else:
                # If no JSON found, return the raw response for error handling
                return {"raw_response": response_text, "parsed": False}

        except Exception as e:
            return {"error": str(e), "success": False}
```

### 2. Open-Source LLM Integration

For privacy and cost considerations, open-source models can be used:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

class OpenSourceRobotLLM:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize with open-source LLM.
        Note: You'll need to handle model access and hardware requirements appropriately.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def plan_task(self, command: str, robot_state: Dict, environment: Dict) -> Dict:
        """Plan task using open-source LLM"""
        prompt = f"""
        [INST] <<SYS>>
        You are a robot task planner. Convert the following natural language command into a sequence of robot actions.

        Robot State: {json.dumps(robot_state)}
        Environment: {json.dumps(environment)}

        Respond in JSON format with task breakdown, potential issues, and success criteria.
        <</SYS>>

        Command: {command}

        Provide a detailed plan in JSON format: [/INST]
        """

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_content = response[json_start:json_end]
                return json.loads(json_content)
            else:
                return {"raw_response": response, "parsed": False}
        except json.JSONDecodeError:
            return {"raw_response": response, "parsed": False, "error": "Could not parse JSON"}
```

## Function Calling for Robotics APIs

### OpenAI Function Calling

LLMs can be enhanced with function calling to directly interact with robot APIs:

```python
import json
from typing import Dict, Any

class RobotFunctionCaller:
    def __init__(self):
        self.functions = {
            "move_to_location": {
                "name": "move_to_location",
                "description": "Move the robot to a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate"},
                        "y": {"type": "number", "description": "Y coordinate"},
                        "z": {"type": "number", "description": "Z coordinate"},
                        "orientation": {"type": "number", "description": "Orientation in radians"}
                    },
                    "required": ["x", "y"]
                }
            },
            "pick_object": {
                "name": "pick_object",
                "description": "Pick up an object",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {"type": "string", "description": "Name of the object to pick"},
                        "location": {"type": "string", "description": "Where to find the object"}
                    },
                    "required": ["object_name"]
                }
            },
            "place_object": {
                "name": "place_object",
                "description": "Place an object at a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {"type": "string", "description": "Name of the object to place"},
                        "location": {"type": "string", "description": "Where to place the object"}
                    },
                    "required": ["object_name", "location"]
                }
            },
            "detect_objects": {
                "name": "detect_objects",
                "description": "Detect objects in the environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "string", "description": "Area to scan for objects"}
                    }
                }
            }
        }

    def call_robot_functions(self, command: str, llm_interface) -> Dict:
        """Use LLM with function calling for robot planning"""
        messages = [
            {
                "role": "system",
                "content": "You are a robot task planner. Use available functions to plan and execute tasks."
            },
            {
                "role": "user",
                "content": command
            }
        ]

        # First, let LLM decide which functions to call
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            functions=list(self.functions.values()),
            function_call="auto",
            temperature=0.3
        )

        response_message = response.choices[0].message

        # If the model wants to call a function
        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_args = json.loads(response_message["function_call"]["arguments"])

            # Execute the function
            result = self.execute_function(function_name, function_args)

            # Add the result to the conversation
            messages.append(response_message)
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(result)
            })

            # Get the final response
            second_response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.3
            )

            return second_response.choices[0].message

        return response_message

    def execute_function(self, function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute robot function and return result"""
        if function_name == "move_to_location":
            return self._move_to_location(**args)
        elif function_name == "pick_object":
            return self._pick_object(**args)
        elif function_name == "place_object":
            return self._place_object(**args)
        elif function_name == "detect_objects":
            return self._detect_objects(**args)
        else:
            return {"error": f"Unknown function: {function_name}"}

    def _move_to_location(self, x: float, y: float, z: float = 0.0, orientation: float = 0.0) -> Dict[str, Any]:
        """Simulate moving to location"""
        # In a real implementation, this would interface with navigation stack
        return {
            "success": True,
            "message": f"Moved to location ({x}, {y}, {z}) with orientation {orientation}",
            "actual_position": {"x": x, "y": y, "z": z}
        }

    def _pick_object(self, object_name: str, location: str = None) -> Dict[str, Any]:
        """Simulate picking up an object"""
        # In a real implementation, this would interface with manipulation stack
        return {
            "success": True,
            "message": f"Picked up {object_name}",
            "object_status": "held"
        }

    def _place_object(self, object_name: str, location: str) -> Dict[str, Any]:
        """Simulate placing an object"""
        # In a real implementation, this would interface with manipulation stack
        return {
            "success": True,
            "message": f"Placed {object_name} at {location}",
            "object_status": "placed"
        }

    def _detect_objects(self, area: str = "current_view") -> Dict[str, Any]:
        """Simulate object detection"""
        # In a real implementation, this would interface with perception stack
        return {
            "success": True,
            "objects_detected": [
                {"name": "cup", "confidence": 0.95, "position": {"x": 1.0, "y": 2.0, "z": 0.8}},
                {"name": "book", "confidence": 0.89, "position": {"x": 1.2, "y": 2.1, "z": 0.85}}
            ],
            "area_scanned": area
        }
```

## ROS 2 Integration

### LLM Planning Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import json
import openai

class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Initialize LLM interface
        api_key = self.declare_parameter('openai_api_key', '').get_parameter_value().string_value
        if not api_key:
            self.get_logger().error("OpenAI API key not provided")
            return

        openai.api_key = api_key
        self.llm_interface = RobotLLMInterface(api_key)

        # Subscriptions
        self.command_sub = self.create_subscription(
            String,
            'voice_commands',
            self.command_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.robot_pose_sub = self.create_subscription(
            Pose,
            'robot_pose',
            self.robot_pose_callback,
            10
        )

        # Publishers
        self.plan_pub = self.create_publisher(String, 'robot_plan', 10)
        self.status_pub = self.create_publisher(String, 'llm_status', 10)

        # Robot state tracking
        self.current_joint_state = None
        self.current_pose = None

        self.get_logger().info('LLM Planning Node initialized')

    def command_callback(self, msg: String):
        """Process voice command and generate plan"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Get current robot state
        robot_state = self.get_current_robot_state()
        environment = self.get_environment_context()

        # Generate plan using LLM
        plan = self.llm_interface.plan_task(command, robot_state, environment)

        if "error" not in plan:
            # Publish the plan
            plan_msg = String()
            plan_msg.data = json.dumps(plan)
            self.plan_pub.publish(plan_msg)

            self.get_logger().info(f'Generated plan: {plan}')
        else:
            self.get_logger().error(f'Plan generation failed: {plan["error"]}')
            status_msg = String()
            status_msg.data = f"Error: {plan['error']}"
            self.status_pub.publish(status_msg)

    def joint_state_callback(self, msg: JointState):
        """Update joint state"""
        self.current_joint_state = msg

    def robot_pose_callback(self, msg: Pose):
        """Update robot pose"""
        self.current_pose = msg

    def get_current_robot_state(self) -> Dict:
        """Get current robot state"""
        state = {
            "timestamp": self.get_clock().now().to_msg().stamp.sec,
            "position": {},
            "joints": {},
            "battery_level": 100,  # Would come from battery topic
            "capabilities": ["navigation", "manipulation", "perception"]
        }

        if self.current_pose:
            state["position"] = {
                "x": self.current_pose.position.x,
                "y": self.current_pose.position.y,
                "z": self.current_pose.position.z,
                "orientation": {
                    "x": self.current_pose.orientation.x,
                    "y": self.current_pose.orientation.y,
                    "z": self.current_pose.orientation.z,
                    "w": self.current_pose.orientation.w
                }
            }

        if self.current_joint_state:
            state["joints"] = dict(zip(
                self.current_joint_state.name,
                self.current_joint_state.position
            ))

        return state

    def get_environment_context(self) -> Dict:
        """Get environment context"""
        # In a real system, this would come from:
        # - Static map
        # - Dynamic object tracking
        # - Sensor data
        # - Human presence detection
        return {
            "known_locations": {
                "kitchen": {"x": 5.0, "y": 3.0},
                "living_room": {"x": 2.0, "y": 1.0},
                "bedroom": {"x": 8.0, "y": 2.0}
            },
            "recently_detected_objects": [],
            "obstacles": [],  # Would come from costmap
            "navigation_status": "ready"
        }

def main(args=None):
    rclpy.init(args=args)
    llm_planning_node = LLMPlanningNode()

    try:
        rclpy.spin(llm_planning_node)
    except KeyboardInterrupt:
        pass
    finally:
        llm_planning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Planning Techniques

### Hierarchical Task Networks (HTN)

```python
class HTNPlanner:
    def __init__(self):
        self.primitive_actions = {
            'navigate_to': self._navigate_to,
            'pick_up': self._pick_up,
            'place_down': self._place_down,
            'detect_object': self._detect_object
        }

        self.complex_tasks = {
            'fetch_object': self._decompose_fetch_object,
            'clean_table': self._decompose_clean_table,
            'serve_drink': self._decompose_serve_drink
        }

    def decompose_task(self, task_name: str, params: Dict) -> List[Dict]:
        """Decompose high-level task into primitive actions"""
        if task_name in self.complex_tasks:
            return self.complex_tasks[task_name](params)
        elif task_name in self.primitive_actions:
            return [{'action': task_name, 'params': params}]
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def _decompose_fetch_object(self, params: Dict) -> List[Dict]:
        """Decompose fetch object task"""
        object_name = params['object_name']
        destination = params['destination']

        return [
            {'action': 'detect_object', 'params': {'target': object_name}},
            {'action': 'navigate_to', 'params': {'location': f'{object_name}_location'}},
            {'action': 'pick_up', 'params': {'object': object_name}},
            {'action': 'navigate_to', 'params': {'location': destination}},
            {'action': 'place_down', 'params': {'object': object_name, 'location': destination}}
        ]

    def _decompose_clean_table(self, params: Dict) -> List[Dict]:
        """Decompose clean table task"""
        table_location = params['table_location']

        return [
            {'action': 'navigate_to', 'params': {'location': table_location}},
            {'action': 'detect_object', 'params': {'target': 'any_object_on_table'}},
            # This would continue with pickup/dropoff cycles
        ]

    def _decompose_serve_drink(self, params: Dict) -> List[Dict]:
        """Decompose serve drink task"""
        drink_type = params.get('drink_type', 'water')
        customer_location = params['customer_location']

        return [
            {'action': 'navigate_to', 'params': {'location': 'kitchen'}},
            {'action': 'detect_object', 'params': {'target': drink_type}},
            {'action': 'pick_up', 'params': {'object': f'{drink_type}_container'}},
            {'action': 'navigate_to', 'params': {'location': customer_location}},
            {'action': 'place_down', 'params': {'object': f'{drink_type}_container', 'location': 'table_near_customer'}},
            {'action': 'utter', 'params': {'text': 'Here is your drink!'}}
        ]

    def _navigate_to(self, params: Dict):
        """Primitive navigation action"""
        # Interface with Nav2
        pass

    def _pick_up(self, params: Dict):
        """Primitive pick up action"""
        # Interface with manipulation stack
        pass

    def _place_down(self, params: Dict):
        """Primitive place down action"""
        # Interface with manipulation stack
        pass

    def _detect_object(self, params: Dict):
        """Primitive object detection"""
        # Interface with perception stack
        pass
```

## Prompt Engineering for Robotics

### Effective Prompts for Robot Planning

```python
class RobotPlanningPrompts:
    @staticmethod
    def create_task_planning_prompt(command: str, robot_state: Dict, environment: Dict) -> str:
        """Create effective prompt for task planning"""
        return f"""
        You are an expert robot task planner. Your job is to break down human commands into specific, executable robot actions.

        ROBOT CAPABILITIES:
        - Navigation: Can move to specific coordinates (x, y, z) with orientation
        - Manipulation: Can pick up, place, grasp objects
        - Perception: Can detect and recognize objects, people, locations
        - Communication: Can speak, listen, display information

        CURRENT STATE:
        Position: ({robot_state.get('position', {}).get('x', 0)}, {robot_state.get('position', {}).get('y', 0)})
        Battery: {robot_state.get('battery_level', 100)}%
        Connected: True
        Available actions: {', '.join(robot_state.get('capabilities', []))}

        ENVIRONMENT:
        Known locations: {list(environment.get('known_locations', {}).keys())}
        Recent detections: {environment.get('recently_detected_objects', [])}
        Obstacles: {environment.get('obstacles', [])}

        COMMAND: "{command}"

        INSTRUCTIONS:
        1. Analyze the command for specific goals
        2. Consider the robot's current state and environment
        3. Break down into sequential, executable actions
        4. Include error handling and validation steps
        5. Consider safety and feasibility

        OUTPUT FORMAT:
        {{
            "analysis": "Brief analysis of the command",
            "plan": [
                {{
                    "step": 1,
                    "action": "action_type",
                    "parameters": {{"param1": "value1"}},
                    "description": "What this step does",
                    "expected_outcome": "How to verify success",
                    "error_handling": "What to do if this fails"
                }}
            ],
            "estimated_duration": "Time estimate in seconds",
            "resources_needed": ["list", "of", "required", "resources"],
            "potential_risks": ["risk1", "risk2"]
        }}

        Respond ONLY in valid JSON format:
        """

    @staticmethod
    def create_error_recovery_prompt(error: str, attempted_action: Dict, robot_state: Dict) -> str:
        """Create prompt for error recovery"""
        return f"""
        The robot encountered an error during task execution:

        ERROR: {error}
        ATTEMPTED ACTION: {attempted_action}
        CURRENT STATE: {robot_state}

        PROVIDE RECOVERY PLAN:
        1. Diagnose the likely cause of the error
        2. Suggest immediate recovery actions
        3. Propose alternative approaches
        4. Indicate when to abort and ask for human help

        FORMAT: {{
            "diagnosis": "Likely cause of error",
            "immediate_recovery": ["action1", "action2"],
            "alternative_approach": "Different way to achieve goal",
            "abort_conditions": ["conditions", "to", "stop", "trying"],
            "human_help_needed": "When to ask for assistance"
        }}
        """
```

## Safety and Validation

### Plan Validation System

```python
class PlanValidator:
    def __init__(self):
        self.safety_rules = [
            self._check_navigation_safety,
            self._check_manipulation_safety,
            self._check_resource_availability,
            self._check_feasibility
        ]

    def validate_plan(self, plan: Dict, robot_state: Dict, environment: Dict) -> Dict:
        """Validate plan for safety and feasibility"""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }

        for rule in self.safety_rules:
            result = rule(plan, robot_state, environment)
            if not result["valid"]:
                validation_result["is_valid"] = False
                validation_result["issues"].extend(result["issues"])
            validation_result["warnings"].extend(result["warnings"])
            validation_result["suggestions"].extend(result["suggestions"])

        return validation_result

    def _check_navigation_safety(self, plan: Dict, robot_state: Dict, environment: Dict) -> Dict:
        """Check if navigation plan is safe"""
        issues = []
        warnings = []
        suggestions = []

        for step in plan.get("plan", []):
            if step.get("action") == "navigate_to":
                target = step.get("parameters", {})
                x, y = target.get("x"), target.get("y")

                # Check if target is in known safe areas
                if not self._is_safe_navigation_target(x, y, environment):
                    issues.append(f"Navigation to ({x}, {y}) may be unsafe")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions
        }

    def _is_safe_navigation_target(self, x: float, y: float, environment: Dict) -> bool:
        """Check if navigation target is safe"""
        # Check against known obstacles
        obstacles = environment.get("obstacles", [])
        for obstacle in obstacles:
            obs_x, obs_y = obstacle.get("x", 0), obstacle.get("y", 0)
            distance = ((x - obs_x)**2 + (y - obs_y)**2)**0.5
            if distance < 0.5:  # 50cm safety margin
                return False

        # Check if within map boundaries
        # (would check against map in real implementation)

        return True

    def _check_manipulation_safety(self, plan: Dict, robot_state: Dict, environment: Dict) -> Dict:
        """Check if manipulation plan is safe"""
        # Implementation would check:
        # - Reachability
        # - Object properties (weight, fragility)
        # - Collision avoidance
        pass

    def _check_resource_availability(self, plan: Dict, robot_state: Dict, environment: Dict) -> Dict:
        """Check if required resources are available"""
        # Implementation would check:
        # - Battery level for planned duration
        # - Required tools/accessories
        # - Available time before deadlines
        pass

    def _check_feasibility(self, plan: Dict, robot_state: Dict, environment: Dict) -> Dict:
        """Check if plan is technically feasible"""
        # Implementation would check:
        # - Robot capabilities vs required actions
        # - Environmental constraints
        # - Time feasibility
        pass
```

## Performance Optimization

### Caching and Optimization

```python
import functools
import hashlib
import time
from typing import Callable, Any

class OptimizedLLMInterface:
    def __init__(self, api_key: str, cache_size: int = 1000):
        openai.api_key = api_key
        self.cache = {}
        self.cache_order = []  # For LRU eviction
        self.cache_size = cache_size

    def cached_plan_task(self, command: str, robot_state: Dict, environment: Dict) -> Dict:
        """Plan task with caching to reduce API calls"""
        # Create cache key from inputs
        cache_key = self._create_cache_key(command, robot_state, environment)

        # Check cache first
        if cache_key in self.cache:
            self.cache_order.remove(cache_key)  # Remove from current position
            self.cache_order.append(cache_key)  # Move to end (most recent)
            return self.cache[cache_key]

        # Generate new plan
        plan = self._generate_plan(command, robot_state, environment)

        # Add to cache
        self._add_to_cache(cache_key, plan)

        return plan

    def _create_cache_key(self, command: str, robot_state: Dict, environment: Dict) -> str:
        """Create unique cache key"""
        combined = f"{command}|{hash(str(sorted(robot_state.items())))}|{hash(str(sorted(environment.items())))}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _add_to_cache(self, key: str, value: Dict):
        """Add to cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = value
        self.cache_order.append(key)

    def _generate_plan(self, command: str, robot_state: Dict, environment: Dict) -> Dict:
        """Generate plan using LLM (this is the actual API call)"""
        # Implementation of the actual LLM call
        # This would be the method from RobotLLMInterface
        pass
```

## Troubleshooting Common Issues

### 1. API Rate Limits
**Problem**: OpenAI API rate limiting
**Solutions**:
- Implement request queuing
- Use caching for repeated commands
- Monitor token usage
- Consider higher-tier plans for production

### 2. Hallucination Issues
**Problem**: LLM generates invalid or impossible actions
**Solutions**:
- Use function calling to constrain outputs
- Implement validation layers
- Use lower temperatures (0.1-0.3)
- Provide clear examples in system prompts

### 3. Context Window Limitations
**Problem**: Large robot states exceed context window
**Solutions**:
- Summarize robot state before sending
- Use retrieval-augmented generation (RAG)
- Implement state compression
- Use streaming for large inputs

### 4. Integration Latency
**Problem**: High latency in robot response
**Solutions**:
- Use smaller, faster models for simple tasks
- Implement asynchronous processing
- Pre-plan common tasks
- Use local models for basic commands

## Best Practices

### 1. Error Handling
- Always implement fallback strategies
- Log LLM interactions for debugging
- Provide human-in-the-loop options
- Validate outputs before execution

### 2. Security
- Secure API keys properly
- Validate all inputs from LLM
- Implement access controls
- Monitor for prompt injection

### 3. Performance
- Cache frequent requests
- Use appropriate model sizes
- Implement request batching
- Monitor and optimize token usage

### 4. Testing
- Test with edge cases
- Validate safety constraints
- Test error recovery paths
- Performance benchmarking

## Exercise

Create a complete LLM integration for your humanoid robot that includes:

1. A robust LLM interface with error handling
2. Function calling integration for direct robot API access
3. A validation system to ensure plan safety
4. Performance optimization techniques
5. ROS 2 integration for real-time command processing
6. A hierarchical planning system for complex tasks

Test your system with various natural language commands and evaluate its ability to generate safe, executable robot plans.