#pragma once
#ifdef USE_METAL
#include "ARBDException.h"
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"
#include <Metal/Metal.h>
#include <cstddef>
#include <cstring>

namespace ARBD {
namespace METAL {
struct Policy {
	static void* allocate(const Resource& resource,
						  size_t bytes,
						  void* queue = nullptr,
						  bool sync = true,
						  MTL::ResourceOptions storage_mode = MTL::ResourceStorageModeShared) {
		// Metal allocation doesn't need queue or sync parameters
		(void)queue;
		(void)sync;
		if (resource.type != ResourceType::METAL) {
			ARBD_Exception(ExceptionType::ValueError,
						   "Metal Policy requires Metal resource, got {}",
						   resource.toString());
		}

		// Get the specific device for this resource
		auto& device_manager = Manager::get_device(resource.id);
		void* ptr = device_manager.allocate_raw(bytes, storage_mode);

		if (!ptr) {
			ARBD_Exception(ExceptionType::RuntimeError,
						   "Failed to allocate {} bytes on Metal device {}",
						   bytes,
						   resource.id);
		}

		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		// Metal deallocation doesn't need queue or sync parameters
		(void)queue;
		(void)sync;
		if (ptr) {
			Manager::deallocate_raw(ptr);
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		MTL::Buffer* mtl_buffer = Manager::get_metal_buffer_from_ptr(const_cast<void*>(device_src));
		if (!mtl_buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "copy_to_host: Invalid buffer pointer");
		}

		auto& device_manager = Manager::get_current_device();
		auto& queue = device_manager.get_next_queue();
		MTL::CommandBuffer* cmd_buffer =
			static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
		MTL::BlitCommandEncoder* blit_encoder = cmd_buffer->blitCommandEncoder();

		if (mtl_buffer->storageMode() == MTL::StorageModeShared) {
			blit_encoder->synchronizeResource(mtl_buffer);
			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
			std::memcpy(host_dst, mtl_buffer->contents(), bytes);
		} else {
			MTL::Buffer* staging_buffer =
				device_manager.metal_device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
			blit_encoder->copyFromBuffer(mtl_buffer, 0, staging_buffer, 0, bytes);
			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
			std::memcpy(host_dst, staging_buffer->contents(), bytes);
			staging_buffer->release();
		}
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		MTL::Buffer* mtl_buffer = Manager::get_metal_buffer_from_ptr(device_dst);
		if (!mtl_buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "copy_from_host: Invalid buffer pointer");
		}

		if (mtl_buffer->storageMode() == MTL::StorageModeShared) {
			std::memcpy(mtl_buffer->contents(), host_src, bytes);

			auto& device_manager = Manager::get_current_device();
			auto& queue = device_manager.get_next_queue();
			MTL::CommandBuffer* cmd_buffer =
				static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
			MTL::BlitCommandEncoder* blit_encoder = cmd_buffer->blitCommandEncoder();
			blit_encoder->synchronizeResource(mtl_buffer);
			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
		} else {
			auto& device_manager = Manager::get_current_device();
			MTL::Buffer* staging_buffer =
				device_manager.metal_device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
			std::memcpy(staging_buffer->contents(), host_src, bytes);

			auto& queue = device_manager.get_next_queue();
			MTL::CommandBuffer* cmd_buffer =
				static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
			MTL::BlitCommandEncoder* blit_encoder = cmd_buffer->blitCommandEncoder();
			blit_encoder->copyFromBuffer(staging_buffer, 0, mtl_buffer, 0, bytes);
			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
			staging_buffer->release();
		}
	}

	static void copy_device_to_device(void* dst, const void* src, size_t bytes) {
		auto& device_manager = Manager::get_current_device();
		auto& queue = device_manager.get_next_queue();
		MTL::CommandBuffer* cmd_buffer =
			static_cast<MTL::CommandBuffer*>(queue.create_command_buffer());
		MTL::BlitCommandEncoder* blit_encoder = cmd_buffer->blitCommandEncoder();

		MTL::Buffer* src_buffer = Manager::get_metal_buffer_from_ptr(const_cast<void*>(src));
		MTL::Buffer* dst_buffer = Manager::get_metal_buffer_from_ptr(dst);

		if (!src_buffer || !dst_buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "copy_device_to_device: Invalid buffer pointer(s)");
		}

		blit_encoder->copyFromBuffer(src_buffer, 0, dst_buffer, 0, bytes);
		blit_encoder->endEncoding();
		cmd_buffer->commit();
		cmd_buffer->waitUntilCompleted();
	}
};
} // namespace METAL
} // namespace ARBD
#endif
