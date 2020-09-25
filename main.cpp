#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CXX14
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>

const int WIDTH = 800;
const int HEIGHT = 600;

const std::string MODEL_PATH = "models/chalet.obj";
const std::string TEXTURE_PATH = "textures/chalet.jpg";

const int MAX_FRAMES_IN_FLIGHT = 3;

const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec3 pos;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        // vk::VertexInputBindingDescription(binding_, stride_, inputRate_)
        return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions()
    {
        // vk::VertexInputAttributeDescription(location_, binding_, format_, offset_)
        return {{0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)},
                {1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)}};
    }

    bool operator==(const Vertex &other) const
    {
        return pos == other.pos && texCoord == other.texCoord;
    }
};

template <>
struct std::hash<Vertex> {
    size_t operator()(Vertex const &vertex) const
    {
        return (std::hash<glm::vec3>()(vertex.pos) ^ (std::hash<glm::vec2>()(vertex.texCoord) << 1)) >> 1;
    }
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class VulkanApplication
{
  public:
    VulkanApplication()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Application", nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
    }

    void run()
    {
        initVulkan();
        mainLoop();
    }

    ~VulkanApplication()
    {
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

  private:
    GLFWwindow *m_window;

    vk::UniqueInstance m_instance;
    vk::DispatchLoaderDynamic m_dldy;
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> m_debugMessenger;
    vk::UniqueSurfaceKHR m_surface;

    vk::PhysicalDevice m_physicalDevice;
    vk::SampleCountFlagBits m_msaaSamples = vk::SampleCountFlagBits::e1;
    vk::UniqueDevice m_device;

    vk::Queue m_graphicsQueue;
    vk::Queue m_presentQueue;

    vk::UniqueSwapchainKHR m_swapChain;
    vk::Format m_swapChainImageFormat;
    vk::Extent2D m_swapChainExtent;
    std::vector<vk::Image> m_swapChainImages;
    std::vector<vk::UniqueImageView> m_swapChainImageViews;
    std::vector<vk::UniqueFramebuffer> m_swapChainFramebuffers;

    vk::UniqueRenderPass m_renderPass;
    vk::UniqueDescriptorSetLayout m_descriptorSetLayout;
    vk::UniquePipelineLayout m_pipelineLayout;
    std::vector<vk::UniquePipeline> m_graphicsPipelines;

    vk::UniqueCommandPool m_commandPool;

    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;

    uint32_t m_mipLevels;

    vk::UniqueDeviceMemory m_colorImageMemory;
    vk::UniqueDeviceMemory m_depthImageMemory;
    vk::UniqueDeviceMemory m_textureImageMemory;
    vk::UniqueDeviceMemory m_vertexBufferMemory;
    vk::UniqueDeviceMemory m_indexBufferMemory;

    vk::UniqueImage m_colorImage;
    vk::UniqueImage m_depthImage;
    vk::UniqueImage m_textureImage;
    vk::UniqueBuffer m_vertexBuffer;
    vk::UniqueBuffer m_indexBuffer;

    vk::UniqueImageView m_colorImageView;
    vk::UniqueImageView m_depthImageView;
    vk::UniqueImageView m_textureImageView;
    vk::UniqueSampler m_textureSampler;

    std::vector<vk::UniqueDeviceMemory> m_uniformBuffersMemory;
    std::vector<vk::UniqueBuffer> m_uniformBuffers;

    vk::UniqueDescriptorPool m_descriptorPool;
    std::vector<vk::DescriptorSet> m_descriptorSets;

    std::vector<vk::CommandBuffer> m_commandBuffers;

    std::vector<vk::UniqueSemaphore> m_imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphores;
    std::vector<vk::UniqueFence> m_inFlightFences;
    size_t m_currentFrame = 0;

    bool m_framebufferResized = false;

    static void framebufferResizeCallback(GLFWwindow *window, int /*width*/, int /*height*/)
    {
        reinterpret_cast<VulkanApplication *>(glfwGetWindowUserPointer(window))->m_framebufferResized = true;
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (glfwWindowShouldClose(m_window) == 0) {
            if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(m_window, 1);
            }
            glfwPollEvents();
            drawFrame();
        }

        m_device->waitIdle();
    }

    void recreateSwapChain()
    {
        int width = 0;
        int height = 0;
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(m_window, &width, &height);
            glfwWaitEvents();
        }

        m_device->waitIdle();
        m_device->freeCommandBuffers(*m_commandPool, m_commandBuffers);

        createSwapChain();
        createImageViews();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createCommandBuffers();
    }

    void drawFrame()
    {
        m_device->waitForFences({*m_inFlightFences[m_currentFrame]}, VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t imageIndex;
        auto result = m_device->acquireNextImageKHR(*m_swapChain, std::numeric_limits<uint64_t>::max(), *m_imageAvailableSemaphores[m_currentFrame], {}, &imageIndex);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        m_device->resetFences({*m_inFlightFences[m_currentFrame]});

        updateUniformBuffer(imageIndex);

        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        // vk::SubmitInfo(waitSemaphoreCount_, pWaitSemaphores_, pWaitDstStageMask_, commandBufferCount_, pCommandBuffers_, signalSemaphoreCount_, pSignalSemaphores_)
        m_graphicsQueue.submit({{1, &*m_imageAvailableSemaphores[m_currentFrame], &waitStage, 1, &m_commandBuffers[imageIndex], 1, &*m_renderFinishedSemaphores[m_currentFrame]}}, *m_inFlightFences[m_currentFrame]);

        auto presentInfo = vk::PresentInfoKHR(1, &*m_renderFinishedSemaphores[m_currentFrame], 1, &*m_swapChain, &imageIndex, nullptr);
        result = m_presentQueue.presentKHR(&presentInfo);

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || m_framebufferResized) {
            m_framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo = {
            glm::rotate(glm::mat4(1), time * glm::radians(90.0f), glm::vec3(0, 0, 1)),
            glm::lookAt(glm::vec3(2, 2, 2), glm::vec3(0, 0, 0), glm::vec3(0, 0, 1)),
            glm::perspective(glm::radians(45.0f), float(WIDTH) / float(HEIGHT), 0.1f, 10.0f)};

        ubo.proj[1][1] *= -1;

        void *data = m_device->mapMemory(*m_uniformBuffersMemory[currentImage], 0, sizeof(ubo), {});
        memcpy(data, &ubo, sizeof(ubo));
        m_device->unmapMemory(*m_uniformBuffersMemory[currentImage]);
    }

    void createSyncObjects()
    {
        m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            m_imageAvailableSemaphores[i] = m_device->createSemaphoreUnique({});
            m_renderFinishedSemaphores[i] = m_device->createSemaphoreUnique({});
            m_inFlightFences[i] = m_device->createFenceUnique({vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void createCommandBuffers()
    {
        // vk::CommandBufferAllocateInfo(commandPool_, level_, commandBufferCount_)
        m_commandBuffers = m_device->allocateCommandBuffers({*m_commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t)m_swapChainFramebuffers.size()});

        for (size_t i = 0; i < m_commandBuffers.size(); i++) {
            // vk::CommandBufferBeginInfo(flags_, pInheritanceInfo_)
            m_commandBuffers[i].begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse, nullptr});

            std::vector<vk::ClearValue> clearValues = {
                vk::ClearColorValue(std::array{0.0f, 0.0f, 0.0f, 1.0f}),
                vk::ClearDepthStencilValue(1, 0),
            };

            // vk::RenderPassBeginInfo(renderPass_, framebuffer_, renderArea_, clearValueCount_, pClearValues_)
            auto renderPassInfo = vk::RenderPassBeginInfo(*m_renderPass, *m_swapChainFramebuffers[i], {{0, 0}, m_swapChainExtent}, clearValues.size(), clearValues.data());
            m_commandBuffers[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);
            m_commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, *m_graphicsPipelines[0]);
            m_commandBuffers[i].bindVertexBuffers(0, *m_vertexBuffer, {0});
            m_commandBuffers[i].bindIndexBuffer(*m_indexBuffer, 0, vk::IndexType::eUint32);
            m_commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_pipelineLayout, 0, m_descriptorSets[i], {});
            setViewportScissor(m_commandBuffers[i]);
            m_commandBuffers[i].drawIndexed(m_indices.size(), 1, 0, 0, 0);
            m_commandBuffers[i].endRenderPass();
            m_commandBuffers[i].end();
        }
    }

    void setViewportScissor(vk::CommandBuffer commandBuffer)
    {
        // Keep initial image ratio
        float ratio = float(WIDTH) / float(HEIGHT);
        float width = m_swapChainExtent.width;
        float height = m_swapChainExtent.height;

        if (width / height > ratio) {
            width = ratio * height;
        } else {
            height = width / ratio;
        }

        float offset_x = (m_swapChainExtent.width - width) / 2;
        float offset_y = (m_swapChainExtent.height - height) / 2;

        // vk::Viewport(x_, y_, width_, height_, minDepth_, maxDepth_)
        commandBuffer.setViewport(0, {{offset_x, offset_y, width, height, 0, 1}});
        // vk::Rect2D(vk::Offset2D(x_, y_), extent_)
        commandBuffer.setScissor(0, {{{0, 0}, m_swapChainExtent}});
    }

    void createDescriptorSets()
    {
        auto layouts = std::vector<vk::DescriptorSetLayout>(m_swapChainImages.size(), *m_descriptorSetLayout);

        // vk::DescriptorSetAllocateInfo(descriptorPool_, descriptorSetCount_, pSetLayouts_)
        m_descriptorSets = m_device->allocateDescriptorSets({*m_descriptorPool, (uint32_t)m_swapChainImages.size(), layouts.data()});

        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            // vk::DescriptorBufferInfo(buffer_, offset_, range_)
            auto bufferInfo = vk::DescriptorBufferInfo(*m_uniformBuffers[i], 0, sizeof(UniformBufferObject));
            // vk::DescriptorImageInfo(sampler_, imageView_, imageLayout_)
            auto imageInfo = vk::DescriptorImageInfo(*m_textureSampler, *m_textureImageView, vk::ImageLayout::eShaderReadOnlyOptimal);

            // vk::WriteDescriptorSet(dstSet_, dstBinding_, dstArrayElement_, descriptorCount_, descriptorType_, pImageInfo_, pBufferInfo_, pTexelBufferView_)
            m_device->updateDescriptorSets(
                {{m_descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo, nullptr},
                 {m_descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo, nullptr, nullptr}},
                {});
        }
    }

    void createDescriptorPool()
    {
        // vk::DescriptorPoolSize(type_, descriptorCount_)
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eUniformBuffer, (uint32_t)m_swapChainImages.size()},
            {vk::DescriptorType::eCombinedImageSampler, (uint32_t)m_swapChainImages.size()}};

        // vk::DescriptorPoolCreateInfo(flags_, maxSets_, poolSizeCount_, pPoolSizes_)
        m_descriptorPool = m_device->createDescriptorPoolUnique({{}, (uint32_t)m_swapChainImages.size(), (uint32_t)poolSizes.size(), poolSizes.data()});
    }

    void createVertexBuffer()
    {
        vk::DeviceSize bufferSize = sizeof(m_vertices[0]) * m_vertices.size();
        vk::UniqueDeviceMemory stagingBufferMemory;
        vk::UniqueBuffer stagingBuffer;

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void *data = m_device->mapMemory(*stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, m_vertices.data(), bufferSize);
        m_device->unmapMemory(*stagingBufferMemory);

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, m_vertexBuffer, m_vertexBufferMemory);
        copyBuffer(*stagingBuffer, *m_vertexBuffer, bufferSize);
    }

    void createIndexBuffer()
    {
        vk::DeviceSize bufferSize = sizeof(m_indices[0]) * m_indices.size();
        vk::UniqueDeviceMemory stagingBufferMemory;
        vk::UniqueBuffer stagingBuffer;

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void *data = m_device->mapMemory(*stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, m_indices.data(), bufferSize);
        m_device->unmapMemory(*stagingBufferMemory);

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, m_indexBuffer, m_indexBufferMemory);
        copyBuffer(*stagingBuffer, *m_indexBuffer, bufferSize);
    }

    void createUniformBuffers()
    {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        m_uniformBuffers.resize(m_swapChainImages.size());
        m_uniformBuffersMemory.resize(m_swapChainImages.size());

        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_uniformBuffers[i], m_uniformBuffersMemory[i]);
        }
    }

    void createUniqueBuffer(vk::DeviceSize size, const vk::BufferUsageFlags &usage, const vk::MemoryPropertyFlags &properties, vk::UniqueBuffer &buffer, vk::UniqueDeviceMemory &bufferMemory)
    {
        // vk::BufferCreateInfo(flags_, size_, usage_, sharingMode_, queueFamilyIndexCount_, pQueueFamilyIndices_)
        buffer = m_device->createBufferUnique({{}, size, usage, vk::SharingMode::eExclusive, 0, nullptr});

        auto memRequirements = m_device->getBufferMemoryRequirements(*buffer);
        uint32_t memoryType = findMemoryType(memRequirements.memoryTypeBits, properties);

        // vk::MemoryAllocateInfo(allocationSize_, memoryTypeIndex_)
        bufferMemory = m_device->allocateMemoryUnique({memRequirements.size, memoryType});
        m_device->bindBufferMemory(*buffer, *bufferMemory, 0);
    }

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
    {
        auto uniqueCommandBuffers = beginSingleTimeCommands();

        // vk::BufferCopy(srcOffset_, dstOffset_, size_)
        uniqueCommandBuffers[0]->copyBuffer(srcBuffer, dstBuffer, {{0, 0, size}});
        endSingleTimeCommands(uniqueCommandBuffers);
    }

    uint32_t findMemoryType(uint32_t typeFilter, const vk::MemoryPropertyFlags &properties)
    {
        auto memProperties = m_physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) != 0 && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createTextureSampler()
    {
        m_textureSampler = m_device->createSamplerUnique(
            {
                {},                               // flags_
                vk::Filter::eLinear,              // magFilter_
                vk::Filter::eLinear,              // minFilter_
                vk::SamplerMipmapMode::eLinear,   // mipmapMode_
                vk::SamplerAddressMode::eRepeat,  // addressModeU_
                vk::SamplerAddressMode::eRepeat,  // addressModeV_
                vk::SamplerAddressMode::eRepeat,  // addressModeW_
                0,                                // mipLodBias_
                VK_TRUE,                          // anisotropyEnable_
                16,                               // maxAnisotropy_
                VK_FALSE,                         // compareEnable_
                vk::CompareOp::eAlways,           // compareOp_
                0,                                // minLod_
                (float)m_mipLevels,               // maxLod_
                vk::BorderColor::eIntOpaqueBlack, // borderColor_
                VK_FALSE                          // unnormalizedCoordinates_
            });
    }

    void createTextureImageView()
    {
        m_textureImageView = createUniqueImageView(*m_textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, m_mipLevels);
    }

    void createTextureImage()
    {
        int texWidth;
        int texHeight;
        int texChannels;
        stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (pixels == nullptr) {
            throw std::runtime_error("failed to load texture image!");
        }

        m_mipLevels = 1 + std::floor(std::log2(std::max(texWidth, texHeight)));

        vk::DeviceSize imageSize = texWidth * texHeight * STBI_rgb_alpha;
        vk::UniqueDeviceMemory stagingBufferMemory;
        vk::UniqueBuffer stagingBuffer;

        createUniqueBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void *data = m_device->mapMemory(*stagingBufferMemory, 0, imageSize, {});
        memcpy(data, pixels, imageSize);
        m_device->unmapMemory(*stagingBufferMemory);
        stbi_image_free(pixels);

        createUniqueImage(texWidth, texHeight, m_mipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, m_textureImage, m_textureImageMemory);
        transitionImageLayout(*m_textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, m_mipLevels);
        copyBufferToImage(*stagingBuffer, *m_textureImage, texWidth, texHeight);
        generateMipmaps(*m_textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, m_mipLevels);
        // transitioned to vk::ImageLayout::eShaderReadOnlyOptimal while generating mipmaps
    }

    void generateMipmaps(vk::Image image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
    {
        if (!(m_physicalDevice.getFormatProperties(imageFormat).optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        auto uniqueCommandBuffers = beginSingleTimeCommands();

        // vk::ImageMemoryBarrier(srcAccessMask_, dstAccessMask_, oldLayout_, newLayout_, srcQueueFamilyIndex_, dstQueueFamilyIndex_, image_, vk::ImageSubresourceRange(aspectMask_, baseMipLevel_, levelCount_, baseArrayLayer_, layerCount_))
        auto barrier = vk::ImageMemoryBarrier({}, {}, {}, {}, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, {vk::ImageAspectFlagBits::eColor, {}, 1, 0, 1});

        int mipWidth = texWidth;
        int mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            uniqueCommandBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

            // vk::ImageBlit(srcSubresource_, srcOffsets_, dstSubresource_, dstOffsets_)
            uniqueCommandBuffers[0]->blitImage(
                image, vk::ImageLayout::eTransferSrcOptimal,
                image, vk::ImageLayout::eTransferDstOptimal,
                {{{vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
                  {{{0, 0, 0}, {mipWidth, mipHeight, 1}}},
                  {vk::ImageAspectFlagBits::eColor, i, 0, 1},
                  {{{0, 0, 0}, {std::max(mipWidth / 2, 1), std::max(mipHeight / 2, 1), 1}}}}},
                vk::Filter::eLinear);

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            uniqueCommandBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

            mipWidth = std::max(mipWidth / 2, 1);
            mipHeight = std::max(mipHeight / 2, 1);
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        uniqueCommandBuffers[0]->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);
        endSingleTimeCommands(uniqueCommandBuffers);
    }

    void createDepthResources()
    {
        vk::Format depthFormat = findDepthFormat();

        createUniqueImage(m_swapChainExtent.width, m_swapChainExtent.height, 1, m_msaaSamples, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, m_depthImage, m_depthImageMemory);
        m_depthImageView = createUniqueImageView(*m_depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
        transitionImageLayout(*m_depthImage, depthFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
    }

    vk::Format findDepthFormat()
    {
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, const vk::FormatFeatureFlags &features)
    {

        for (const auto &format : candidates) {
            auto props = m_physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        throw std::runtime_error("failed to find supported format!");
    }

    void createColorResources()
    {
        createUniqueImage(m_swapChainExtent.width, m_swapChainExtent.height, 1, m_msaaSamples, m_swapChainImageFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, m_colorImage, m_colorImageMemory);
        m_colorImageView = createUniqueImageView(*m_colorImage, m_swapChainImageFormat, vk::ImageAspectFlagBits ::eColor, 1);
        transitionImageLayout(*m_colorImage, m_swapChainImageFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 1);
    }

    void createUniqueImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, const vk::ImageUsageFlags &usage, const vk::MemoryPropertyFlags &properties, vk::UniqueImage &image, vk::UniqueDeviceMemory &imageMemory)
    {
        // vk::ImageCreateInfo(flags_, imageType_, format_, vk::Extent3D(width_, height_, depth_), mipLevels_, arrayLayers_, samples_, tiling_, usage_, sharingMode_, queueFamilyIndexCount_, pQueueFamilyIndices_, initialLayout_)
        image = m_device->createImageUnique({{}, vk::ImageType::e2D, format, {width, height, 1U}, mipLevels, 1, numSamples, tiling, usage, vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined});

        auto memRequirements = m_device->getImageMemoryRequirements(*image);
        uint32_t memoryType = findMemoryType(memRequirements.memoryTypeBits, properties);

        // vk::MemoryAllocateInfo(allocationSize_, memoryTypeIndex_)
        imageMemory = m_device->allocateMemoryUnique({memRequirements.size, memoryType});
        m_device->bindImageMemory(*image, *imageMemory, 0);
    }

    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels)
    {
        auto uniqueCommandBuffers = beginSingleTimeCommands();

        vk::ImageAspectFlags aspectMask;
        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;
        vk::AccessFlags srcAccessMask;
        vk::AccessFlags dstAccessMask;

        if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal && (format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint)) {
            aspectMask |= vk::ImageAspectFlagBits::eStencil;
        } else if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            aspectMask = vk::ImageAspectFlagBits::eDepth;
        } else {
            aspectMask = vk::ImageAspectFlagBits::eColor;
        }

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            srcAccessMask = {};
            dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;

        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;

        } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            srcAccessMask = {};
            dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;

        } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
            srcAccessMask = {};
            dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        // vk::ImageMemoryBarrier(srcAccessMask_, dstAccessMask_, oldLayout_, newLayout_, srcQueueFamilyIndex_, dstQueueFamilyIndex_, image_, vk::ImageSubresourceRange(aspectMask_, baseMipLevel_, levelCount_, baseArrayLayer_, layerCount_))
        uniqueCommandBuffers[0]->pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, {{srcAccessMask, dstAccessMask, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, {aspectMask, 0, mipLevels, 0, 1}}});
        endSingleTimeCommands(uniqueCommandBuffers);
    }

    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
    {
        auto uniqueCommandBuffers = beginSingleTimeCommands();

        // vk::BufferImageCopy(bufferOffset_, bufferRowLength_, bufferImageHeight_, imageSubresource_, imageOffset_, imageExtent_)
        uniqueCommandBuffers[0]->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {{0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {width, height, 1}}});
        endSingleTimeCommands(uniqueCommandBuffers);
    }

    std::vector<vk::UniqueCommandBuffer> beginSingleTimeCommands()
    {
        // vk::CommandBufferAllocateInfo(commandPool_, level_, commandBufferCount_)
        auto uniqueCommandBuffers = m_device->allocateCommandBuffersUnique({*m_commandPool, vk::CommandBufferLevel::ePrimary, 1});

        // vk::CommandBufferBeginInfo(flags_, pInheritanceInfo_)
        uniqueCommandBuffers[0]->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit, nullptr});
        return uniqueCommandBuffers;
    }

    void endSingleTimeCommands(std::vector<vk::UniqueCommandBuffer> &uniqueCommandBuffers)
    {
        uniqueCommandBuffers[0]->end();

        // vk::SubmitInfo(waitSemaphoreCount_, pWaitSemaphores_, pWaitDstStageMask_, commandBufferCount_, pCommandBuffers_, signalSemaphoreCount_, pSignalSemaphores_)
        m_graphicsQueue.submit({{0, nullptr, nullptr, 1, &*uniqueCommandBuffers[0], 0, nullptr}}, nullptr);
        m_graphicsQueue.waitIdle();
    }

    void createCommandPool()
    {
        // vk::CommandPoolCreateInfo(flags_, queueFamilyIndex_)
        m_commandPool = m_device->createCommandPoolUnique({{}, findQueueFamilies(m_physicalDevice).graphicsFamily.value()});
    }

    void createFramebuffers()
    {
        m_swapChainFramebuffers.resize(m_swapChainImageViews.size());
        for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
            // colorAttachment, depthAttachment, colorAttachmentResolve
            std::array attachments = {*m_colorImageView, *m_depthImageView, *m_swapChainImageViews[i]};

            // vk::FramebufferCreateInfo(flags_, renderPass_, attachmentCount_, pAttachments_, width_, height_, layers_)
            m_swapChainFramebuffers[i] = m_device->createFramebufferUnique({{}, *m_renderPass, (uint32_t)attachments.size(), attachments.data(), m_swapChainExtent.width, m_swapChainExtent.height, 1});
        }
    }

    void loadModel()
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices;
        for (const auto &shape : shapes) {
            for (const auto &index : shape.mesh.indices) {
                Vertex vertex = {
                    glm::make_vec3(&attrib.vertices[3 * index.vertex_index]),
                    glm::make_vec2(&attrib.texcoords[2 * index.texcoord_index])};

                vertex.texCoord[1] = 1 - vertex.texCoord[1];

                if (uniqueVertices.count(vertex) == 0) {
                    // cppcheck-suppress stlFindInsert
                    uniqueVertices[vertex] = m_vertices.size();
                    m_vertices.push_back(vertex);
                }
                m_indices.push_back(uniqueVertices[vertex]);
            }
        }
    }

    void createGraphicsPipeline()
    {
        vk::UniqueShaderModule vertShaderModule = createShaderModule("shaders/shader.vert.spv");
        vk::UniqueShaderModule fragShaderModule = createShaderModule("shaders/shader.frag.spv");

        // vk::PipelineShaderStageCreateInfo(flags_, stage_, module_, pName_, pSpecializationInfo_)
        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
            {{}, vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "main", nullptr},
            {{}, vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", nullptr}};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        // vk::PipelineVertexInputStateCreateInfo(flags_, vertexBindingDescriptionCount_, pVertexBindingDescriptions_, vertexAttributeDescriptionCount_, pVertexAttributeDescriptions_)
        auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo({}, 1, &bindingDescription, attributeDescriptions.size(), attributeDescriptions.data());

        // vk::PipelineInputAssemblyStateCreateInfo(flags_, topology_, primitiveRestartEnable_)
        auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);

        // Set dynamic viewport and scissor
        auto viewport = vk::Viewport{};
        auto scissor = vk::Rect2D{};
        auto viewportState = vk::PipelineViewportStateCreateInfo({}, 1, &viewport, 1, &scissor);

        // vk::PipelineRasterizationStateCreateInfo(flags_, depthClampEnable_, rasterizerDiscardEnable_, polygonMode_, cullMode_, frontFace_, depthBiasEnable_, depthBiasConstantFactor_, depthBiasClamp_, depthBiasSlopeFactor_, lineWidth_)
        auto rasterizer = vk::PipelineRasterizationStateCreateInfo({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, VK_FALSE, 0, 0, 0, 1);

        // vk::PipelineMultisampleStateCreateInfo(flags_, rasterizationSamples_, sampleShadingEnable_, minSampleShading_, pSampleMask_, alphaToCoverageEnable_, alphaToOneEnable_)
        auto multisampling = vk::PipelineMultisampleStateCreateInfo({}, m_msaaSamples, VK_FALSE, 0, nullptr, VK_FALSE, VK_FALSE);

        // vk::PipelineDepthStencilStateCreateInfo(flags_, depthTestEnable_, depthWriteEnable_, depthCompareOp_, depthBoundsTestEnable_, stencilTestEnable_, front_, back_, minDepthBounds_, maxDepthBounds_)
        auto depthStencil = vk::PipelineDepthStencilStateCreateInfo({}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess, VK_FALSE, VK_FALSE, {}, {}, 0, 0);

        // vk::PipelineColorBlendAttachmentState(blendEnable_, srcColorBlendFactor_, dstColorBlendFactor_, colorBlendOp_, srcAlphaBlendFactor_, dstAlphaBlendFactor_, alphaBlendOp_, colorWriteMask_)
        auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState(
            VK_FALSE,
            vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
            vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

        // vk::PipelineColorBlendStateCreateInfo(flags_, logicOpEnable_, logicOp_, attachmentCount_, pAttachments_, blendConstants_)
        auto colorBlending = vk::PipelineColorBlendStateCreateInfo({}, VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment, {0, 0, 0, 0});

        // vk::PipelineDynamicStateCreateInfo(flags_, dynamicStateCount_, pDynamicStates_)
        auto dynamicStates = std::vector<vk::DynamicState>{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        auto dynamicStateInfo = vk ::PipelineDynamicStateCreateInfo({}, dynamicStates.size(), dynamicStates.data());

        // vk::PipelineLayoutCreateInfo(flags_, setLayoutCount_, pSetLayouts_, pushConstantRangeCount_, pPushConstantRanges_)
        m_pipelineLayout = m_device->createPipelineLayoutUnique({{}, 1, &*m_descriptorSetLayout, 0, nullptr});

        m_graphicsPipelines = m_device->createGraphicsPipelinesUnique(
            vk::PipelineCache(),
            {{
                {},                            // flags_
                (uint32_t)shaderStages.size(), // stageCount_
                shaderStages.data(),           // pStages_
                &vertexInputInfo,              // pVertexInputState_
                &inputAssembly,                // pInputAssemblyState_
                nullptr,                       // pTessellationState_
                &viewportState,                // pViewportState_
                &rasterizer,                   // pRasterizationState_
                &multisampling,                // pMultisampleState_
                &depthStencil,                 // pDepthStencilState_
                &colorBlending,                // pColorBlendState_
                &dynamicStateInfo,             // pDynamicState_
                *m_pipelineLayout,             // layout_
                *m_renderPass,                 // renderPass_
                0,                             // subpass_
                nullptr,                       // basePipelineHandle_
                -1                             // basePipelineIndex_
            }});
    }

    vk::UniqueShaderModule createShaderModule(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error(std::string("failed to open '") + filename + "'!");
        }

        std::vector<char> buffer(file.tellg());
        file.seekg(0);
        file.read(buffer.data(), buffer.size());
        file.close();

        // vk::ShaderModuleCreateInfo(flags_, codeSize_, pCode_)
        return m_device->createShaderModuleUnique({{}, buffer.size(), reinterpret_cast<const uint32_t *>(buffer.data())});
    }

    void createDescriptorSetLayout()
    {
        // vk::DescriptorSetLayoutBinding(binding_, descriptorType_, descriptorCount_, stageFlags_, pImmutableSamplers_)
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr},
            {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr}};

        // vk::DescriptorSetLayoutCreateInfo(flags_, bindingCount_, pBindings_)
        m_descriptorSetLayout = m_device->createDescriptorSetLayoutUnique({{}, (uint32_t)bindings.size(), bindings.data()});
    }

    void createRenderPass()
    {
        // vk::AttachmentDescription(flags_, format_, samples_, loadOp_, storeOp_, stencilLoadOp_, stencilStoreOp_, initialLayout_, finalLayout_)
        std::vector<vk::AttachmentDescription> attachments = {
            /* colorAttachment*/ {{}, m_swapChainImageFormat, m_msaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal},
            /* depthAttachment*/ {{}, findDepthFormat(), m_msaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
            /* colorAttachmentResolve*/ {{}, m_swapChainImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR}};

        // vk::AttachmentReference(attachment_, layout_)
        std::vector<vk::AttachmentReference> attachmentRefs = {
            {0, vk::ImageLayout::eColorAttachmentOptimal},
            {1, vk::ImageLayout::eDepthStencilAttachmentOptimal},
            {2, vk::ImageLayout::eColorAttachmentOptimal}};

        // vk::SubpassDescription(flags_, pipelineBindPoint_, inputAttachmentCount_, pInputAttachments_, colorAttachmentCount_, pColorAttachments_, pResolveAttachments_, pDepthStencilAttachment_, preserveAttachmentCount_, pPreserveAttachments_)
        auto subpass = vk::SubpassDescription({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &attachmentRefs[0], &attachmentRefs[2], &attachmentRefs[1], 0, nullptr);

        // vk::SubpassDependency(srcSubpass_, dstSubpass_, srcStageMask_, dstStageMask_, srcAccessMask_, dstAccessMask_, dependencyFlags_)
        auto dependency = vk::SubpassDependency(
            VK_SUBPASS_EXTERNAL, 0,
            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {}, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

        // vk::RenderPassCreateInfo(flags_, attachmentCount_, pAttachments_, subpassCount_, pSubpasses_, dependencyCount_, pDependencies_)
        m_renderPass = m_device->createRenderPassUnique({{}, (uint32_t)attachments.size(), attachments.data(), 1, &subpass, 1, &dependency});
    }

    void createImageViews()
    {
        m_swapChainImageViews.resize(m_swapChainImages.size());
        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            m_swapChainImageViews[i] = createUniqueImageView(m_swapChainImages[i], m_swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);
        }
    }

    vk::UniqueImageView createUniqueImageView(vk::Image image, vk::Format format, const vk::ImageAspectFlags &aspectFlags, uint32_t mipLevels)
    {
        // vk::ImageViewCreateInfo(flags_, image_, viewType_, format_, components_, vk::ImageSubresourceRange(aspectMask_, baseMipLevel_, levelCount_, baseArrayLayer_, layerCount_))
        return m_device->createImageViewUnique({{}, image, vk::ImageViewType::e2D, format, {}, {aspectFlags, 0, mipLevels, 0, 1}});
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        auto imageSharingMode = vk::SharingMode::eExclusive;
        uint32_t queueFamilyIndexCount = 0;
        uint32_t *pQueueFamilyIndices = nullptr;

        QueueFamilyIndices queueIndices = findQueueFamilies(m_physicalDevice);

        auto queueFamilyIndices = std::array<uint32_t, 2>{queueIndices.graphicsFamily.value(), queueIndices.presentFamily.value()};
        if (queueIndices.graphicsFamily != queueIndices.presentFamily) {
            imageSharingMode = vk::SharingMode::eConcurrent;
            queueFamilyIndexCount = 2;
            pQueueFamilyIndices = queueFamilyIndices.data();
        }

        m_swapChain = m_device->createSwapchainKHRUnique(
            {
                {},                                             // flags_
                *m_surface,                                     // surface_
                imageCount,                                     // minImageCount_
                surfaceFormat.format,                           // imageFormat_
                surfaceFormat.colorSpace,                       // imageColorSpace_
                extent,                                         // imageExtent_
                1,                                              // imageArrayLayers_
                vk::ImageUsageFlagBits::eColorAttachment,       // imageUsage_
                imageSharingMode,                               // imageSharingMode_
                queueFamilyIndexCount,                          // queueFamilyIndexCount_
                pQueueFamilyIndices,                            // pQueueFamilyIndices_
                swapChainSupport.capabilities.currentTransform, // preTransform_
                vk::CompositeAlphaFlagBitsKHR::eOpaque,         // compositeAlpha_
                presentMode,                                    // presentMode_
                VK_TRUE,                                        // clipped_
                nullptr                                         // oldSwapchain_
            });

        m_swapChainImages = m_device->getSwapchainImagesKHR(*m_swapChain);
        m_swapChainImageFormat = surfaceFormat.format;
        m_swapChainExtent = extent;
    }

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        auto it = std::find_if(availableFormats.cbegin(), availableFormats.cend(),
                               [](const auto &format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });

        if (it != availableFormats.cend()) {
            return *it;
        }
        return availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
    {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;
        for (const auto &availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
            if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = availablePresentMode;
            }
        }
        return bestMode;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        int width;
        int height;
        glfwGetFramebufferSize(m_window, &width, &height);

        return {std::clamp((uint32_t)width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp((uint32_t)height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices queueIndices = findQueueFamilies(m_physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {queueIndices.graphicsFamily.value(), queueIndices.presentFamily.value()};

        float queuePriority = 1;
        queueCreateInfos.reserve(uniqueQueueFamilies.size());

        // vk::DeviceQueueCreateInfo(flags_, queueFamilyIndex_, queueCount_, pQueuePriorities_)
        std::transform(uniqueQueueFamilies.cbegin(), uniqueQueueFamilies.cend(), std::back_inserter(queueCreateInfos),
                       [&](const auto &queueFamily) { return vk::DeviceQueueCreateInfo({}, queueFamily, 1, &queuePriority); });

        vk::PhysicalDeviceFeatures deviceFeatures = vk::PhysicalDeviceFeatures().setSamplerAnisotropy(VK_TRUE);

        // vk::DeviceCreateInfo(flags_, queueCreateInfoCount_, pQueueCreateInfos_, enabledLayerCount_, ppEnabledLayerNames, enabledExtensionCount_, ppEnabledExtensionNames_, pEnabledFeatures_)
        m_device = m_physicalDevice.createDeviceUnique({{}, (uint32_t)queueCreateInfos.size(), queueCreateInfos.data(), 0, nullptr, (uint32_t)deviceExtensions.size(), deviceExtensions.data(), &deviceFeatures});

        m_graphicsQueue = m_device->getQueue(queueIndices.graphicsFamily.value(), 0);
        m_presentQueue = m_device->getQueue(queueIndices.presentFamily.value(), 0);
    }

    void pickPhysicalDevice()
    {
        auto devices = m_instance->enumeratePhysicalDevices();
        auto it = std::find_if(devices.cbegin(), devices.cend(), [&](const auto &device) { return isDeviceSuitable(device); });
        if (it == devices.cend()) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        m_physicalDevice = *it;
        m_msaaSamples = getMaxUsableSampleCount();
    }

    vk::SampleCountFlagBits getMaxUsableSampleCount()
    {
        auto limits = m_physicalDevice.getProperties().limits;
        vk::SampleCountFlags counts = limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;

        if (counts & vk::SampleCountFlagBits::e64) {
            return vk::SampleCountFlagBits::e64;
        }
        if (counts & vk::SampleCountFlagBits::e32) {
            return vk::SampleCountFlagBits::e32;
        }
        if (counts & vk::SampleCountFlagBits::e16) {
            return vk::SampleCountFlagBits::e16;
        }
        if (counts & vk::SampleCountFlagBits::e8) {
            return vk::SampleCountFlagBits::e8;
        }
        if (counts & vk::SampleCountFlagBits::e4) {
            return vk::SampleCountFlagBits::e4;
        }
        if (counts & vk::SampleCountFlagBits::e2) {
            return vk::SampleCountFlagBits::e2;
        }
        return vk::SampleCountFlagBits::e1;
    }

    bool isDeviceSuitable(const vk::PhysicalDevice &device)
    {
        if (findQueueFamilies(device).isComplete() && checkDeviceExtensionSupport(device)) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            return !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty() && device.getFeatures().samplerAnisotropy != 0;
        }
        return false;
    }

    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice &device)
    {
        return {device.getSurfaceCapabilitiesKHR(*m_surface), device.getSurfaceFormatsKHR(*m_surface), device.getSurfacePresentModesKHR(*m_surface)};
    }

    static bool checkDeviceExtensionSupport(const vk::PhysicalDevice &device)
    {
        std::set<std::string> requiredExtensions(deviceExtensions.cbegin(), deviceExtensions.cend());
        for (const auto &extension : device.enumerateDeviceExtensionProperties()) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice &device)
    {
        QueueFamilyIndices queueIndices;

        int i = 0;
        for (const auto &queueFamily : device.getQueueFamilyProperties()) {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                queueIndices.graphicsFamily = i;
            }

            vk::Bool32 presentSupport = 0;
            device.getSurfaceSupportKHR(i, *m_surface, &presentSupport);

            if (queueFamily.queueCount > 0 && presentSupport != 0) {
                queueIndices.presentFamily = i;
            }

            if (queueIndices.isComplete()) {
                break;
            }
            i++;
        }
        return queueIndices;
    }

    void createSurface()
    {
        VkSurfaceKHR surfaceTmp;
        if (glfwCreateWindowSurface(*m_instance, m_window, nullptr, &surfaceTmp) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        // Set the instance as the allocator for the surface for correct order of destruction
        m_surface = vk::UniqueSurfaceKHR(surfaceTmp, *m_instance);
    }

    void setupDebugMessenger()
    {
        if (enableValidationLayers) {
            // vk::DebugUtilsMessengerCreateInfoEXT(flags_, messageSeverity_, messageType_, pfnUserCallback_)
            m_debugMessenger = m_instance->createDebugUtilsMessengerEXTUnique(
                {{},
                 vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                 vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                 debugCallback},
                nullptr,
                m_dldy);
        }
    }

    void createInstance()
    {
        // vk::ApplicationInfo(pApplicationName_, applicationVersion_, pEngineName_, engineVersion_, apiVersion_)
        auto appInfo = vk::ApplicationInfo("Vulkan Application", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_1);

        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        uint32_t enabledLayerCount;
        const char *const *ppEnabledLayerNames;
        if (enableValidationLayers) {
            enabledLayerCount = validationLayers.size();
            ppEnabledLayerNames = validationLayers.data();
        } else {
            enabledLayerCount = 0;
            ppEnabledLayerNames = nullptr;
        }

        auto extensions = getRequiredExtensions();

        // vk::InstanceCreateInfo(flags_, pApplicationInfo_, enabledLayerCount_, ppEnabledLayerNames_, enabledExtensionCount_, ppEnabledExtensionNames_)
        m_instance = vk::createInstanceUnique({{}, &appInfo, enabledLayerCount, ppEnabledLayerNames, (uint32_t)extensions.size(), extensions.data()});
        m_dldy.init(*m_instance);
    }

    static bool checkValidationLayerSupport()
    {
        std::set<std::string> requiredLayers(validationLayers.cbegin(), validationLayers.cend());
        for (const auto &layerProperties : vk::enumerateInstanceLayerProperties()) {
            requiredLayers.erase(layerProperties.layerName);
        }
        return requiredLayers.empty();
    }

    static std::vector<const char *> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT /*messageType*/, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void * /*pUserData*/)
    {
        std::cerr << "validation layer: (severity: 0x" << std::setfill('0') << std::setw(4) << std::hex << messageSeverity << ") " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

int main()
{
    VulkanApplication app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
