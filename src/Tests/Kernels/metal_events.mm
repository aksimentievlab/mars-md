#ifdef USE_METAL

#include "../catch_boiler.h"

using namespace ARBD;
using namespace ARBD::BACKEND;

TEST_CASE("Metal Event Creation and Basic Operations", "[metal][events]") {
    // Initialize Metal manager
    ARBD::METAL::METALManager::init();
    ARBD::METAL::METALManager::load_info();
    
    auto& device = ARBD::METAL::METALManager::get_current_device();
    auto& queue = ARBD::METAL::METALManager::get_current_queue();
    
    SECTION("Create Metal Event from command buffer") {
        // Create a command buffer
        void* cmd_buffer = queue.create_command_buffer();
        REQUIRE(cmd_buffer != nullptr);
        
        // Create Metal event
        ARBD::METAL::Event metal_event(cmd_buffer);
        
        // Create unified event
        Resource metal_resource(ResourceType::METAL, 0);
        Event unified_event(metal_event, metal_resource);
        
        REQUIRE(unified_event.is_valid());
        REQUIRE(unified_event.get_resource().type == ResourceType::METAL);
    }
    
    SECTION("Metal Event wait and completion") {
        // Create a command buffer
        void* cmd_buffer = queue.create_command_buffer();
        ARBD::METAL::Event metal_event(cmd_buffer);
        
        // Create unified event
        Resource metal_resource(ResourceType::METAL, 0);
        Event unified_event(metal_event, metal_resource);
        
        // Commit the command buffer
        metal_event.commit();
        
        // Test completion status
        REQUIRE(unified_event.is_valid());
        
        // Wait for completion
        unified_event.wait();
        
        // Check if complete
        REQUIRE(unified_event.is_complete());
    }
    
    SECTION("EventList with Metal events") {
        EventList event_list;
        
        // Create multiple Metal events
        for (int i = 0; i < 3; ++i) {
            void* cmd_buffer = queue.create_command_buffer();
            ARBD::METAL::Event metal_event(cmd_buffer);
            
            // Commit the command buffer before adding to event list
            metal_event.commit();
            
            Resource metal_resource(ResourceType::METAL, 0);
            Event unified_event(metal_event, metal_resource);
            
            event_list.add(unified_event);
        }
        
        REQUIRE(event_list.get_events().size() == 3);
        
        // Test Metal-specific extraction
        auto metal_events = event_list.get_metal_events();
        REQUIRE(metal_events.size() == 3);
        
        // Wait for all events
        event_list.wait_all();
        
        // Check if all are complete
        REQUIRE(event_list.all_complete());
    }
    
    // Cleanup
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal Event Timing", "[metal][events][timing]") {
    // Initialize Metal manager
    ARBD::METAL::METALManager::init();
    ARBD::METAL::METALManager::load_info();
    
    auto& queue = ARBD::METAL::METALManager::get_current_queue();
    
    SECTION("Event timing functionality") {
        // Create a command buffer
        void* cmd_buffer = queue.create_command_buffer();
        ARBD::METAL::Event metal_event(cmd_buffer);
        
        // Commit and wait
        metal_event.commit();
        metal_event.wait();
        
        // Check timing
        auto execution_time = metal_event.get_execution_time();
        REQUIRE(execution_time.count() >= 0);
    }
    
    // Cleanup
    ARBD::METAL::METALManager::finalize();
}

#endif // USE_METAL 