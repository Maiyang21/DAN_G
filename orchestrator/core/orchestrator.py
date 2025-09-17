"""
DAN_G Orchestrator Agent - Core Orchestrator Class

This module contains the main orchestrator class that coordinates all system operations
using Deepseek R1 LLM for intelligent decision making and module coordination.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from .decision_engine import DecisionEngine
from .module_coordinator import ModuleCoordinator
from ..llm.deepseek_r1 import DeepseekR1Integration


class DAN_GOrchestrator:
    """
    DAN_G Orchestrator Agent - Central intelligence of the APOS system.
    
    This class coordinates all system operations, makes intelligent decisions
    about when to invoke specific modules, and manages the overall system state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DAN_G Orchestrator.
        
        Args:
            config: Configuration dictionary containing system settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.decision_engine = DecisionEngine(config.get('decision_engine', {}))
        self.module_coordinator = ModuleCoordinator(config.get('modules', {}))
        self.llm_integration = DeepseekR1Integration(config.get('llm', {}))
        
        # System state
        self.system_state = {
            'status': 'initializing',
            'last_update': datetime.now(),
            'active_modules': [],
            'pending_operations': [],
            'performance_metrics': {}
        }
        
        self.logger.info("DAN_G Orchestrator initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the orchestrator and all modules.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing DAN_G Orchestrator...")
            
            # Initialize LLM integration
            await self.llm_integration.initialize()
            
            # Initialize module coordinator
            await self.module_coordinator.initialize()
            
            # Initialize decision engine
            await self.decision_engine.initialize()
            
            # Update system state
            self.system_state['status'] = 'ready'
            self.system_state['last_update'] = datetime.now()
            
            self.logger.info("DAN_G Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            self.system_state['status'] = 'error'
            return False
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming request using LLM-powered decision making.
        
        Args:
            request: Request dictionary containing operation details
            
        Returns:
            Dict containing the response and any results
        """
        try:
            self.logger.info(f"Processing request: {request.get('type', 'unknown')}")
            
            # Analyze request using LLM
            analysis = await self.llm_integration.analyze_request(request)
            
            # Make decision about which modules to invoke
            decision = await self.decision_engine.make_decision(analysis, self.system_state)
            
            # Execute the decision
            result = await self._execute_decision(decision, request)
            
            # Update system state
            self._update_system_state(decision, result)
            
            return {
                'success': True,
                'result': result,
                'decision': decision,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def invoke_module(self, module_name: str, operation: str, 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a specific module with given parameters.
        
        Args:
            module_name: Name of the module to invoke
            operation: Operation to perform
            parameters: Parameters for the operation
            
        Returns:
            Dict containing the module response
        """
        try:
            self.logger.info(f"Invoking module {module_name} with operation {operation}")
            
            # Check if module is available
            if not await self.module_coordinator.is_module_available(module_name):
                raise ValueError(f"Module {module_name} is not available")
            
            # Invoke the module
            result = await self.module_coordinator.invoke_module(
                module_name, operation, parameters
            )
            
            # Update system state
            self.system_state['active_modules'].append({
                'module': module_name,
                'operation': operation,
                'start_time': datetime.now(),
                'status': 'completed'
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error invoking module {module_name}: {e}")
            raise
    
    async def _execute_decision(self, decision: Dict[str, Any], 
                              request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a decision made by the decision engine.
        
        Args:
            decision: Decision dictionary from decision engine
            request: Original request
            
        Returns:
            Dict containing execution results
        """
        results = {}
        
        # Execute each action in the decision
        for action in decision.get('actions', []):
            module_name = action.get('module')
            operation = action.get('operation')
            parameters = action.get('parameters', {})
            
            try:
                # Invoke the module
                result = await self.invoke_module(module_name, operation, parameters)
                results[module_name] = result
                
            except Exception as e:
                self.logger.error(f"Failed to execute action {action}: {e}")
                results[module_name] = {'error': str(e)}
        
        return results
    
    def _update_system_state(self, decision: Dict[str, Any], result: Dict[str, Any]):
        """
        Update the system state based on decision and results.
        
        Args:
            decision: Decision that was made
            result: Results from executing the decision
        """
        self.system_state['last_update'] = datetime.now()
        self.system_state['pending_operations'] = [
            op for op in self.system_state['pending_operations']
            if op['id'] not in [action.get('id') for action in decision.get('actions', [])]
        ]
        
        # Update performance metrics
        self.system_state['performance_metrics'].update({
            'total_requests': self.system_state['performance_metrics'].get('total_requests', 0) + 1,
            'successful_requests': self.system_state['performance_metrics'].get('successful_requests', 0) + 1,
            'last_decision_time': datetime.now().isoformat()
        })
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            Dict containing system status information
        """
        return {
            'orchestrator': self.system_state,
            'modules': await self.module_coordinator.get_module_status(),
            'llm': await self.llm_integration.get_status(),
            'decision_engine': await self.decision_engine.get_status()
        }
    
    async def shutdown(self):
        """
        Shutdown the orchestrator and clean up resources.
        """
        try:
            self.logger.info("Shutting down DAN_G Orchestrator...")
            
            # Shutdown modules
            await self.module_coordinator.shutdown()
            
            # Shutdown LLM integration
            await self.llm_integration.shutdown()
            
            # Update system state
            self.system_state['status'] = 'shutdown'
            self.system_state['last_update'] = datetime.now()
            
            self.logger.info("DAN_G Orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Example usage
async def main():
    """Example usage of the DAN_G Orchestrator."""
    
    # Configuration
    config = {
        'llm': {
            'model': 'deepseek-r1',
            'api_key': 'your-api-key',
            'temperature': 0.7
        },
        'modules': {
            'forecasting': {
                'enabled': True,
                'timeout': 300
            },
            'analysis': {
                'enabled': True,
                'timeout': 180
            },
            'optimization': {
                'enabled': True,
                'timeout': 600
            }
        },
        'decision_engine': {
            'confidence_threshold': 0.8,
            'max_retries': 3
        }
    }
    
    # Initialize orchestrator
    orchestrator = DAN_GOrchestrator(config)
    
    # Initialize system
    if await orchestrator.initialize():
        print("Orchestrator initialized successfully")
        
        # Example request
        request = {
            'type': 'forecast_request',
            'data': {'targets': ['yield', 'quality'], 'horizon': 7},
            'context': {'process_conditions': 'normal'}
        }
        
        # Process request
        result = await orchestrator.process_request(request)
        print(f"Request processed: {result}")
        
        # Get system status
        status = await orchestrator.get_system_status()
        print(f"System status: {status}")
        
        # Shutdown
        await orchestrator.shutdown()
    else:
        print("Failed to initialize orchestrator")


if __name__ == "__main__":
    asyncio.run(main())
