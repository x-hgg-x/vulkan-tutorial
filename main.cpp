#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <stddef.h>
#include <stdexcept>

const int WIDTH = 800;
const int HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 1;

const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
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
                {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)},
                {2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)}};
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5, -0.5, 0}, {1, 0, 0}, {1, 0}},
    {{0.5, -0.5, 0}, {0, 1, 0}, {0, 0}},
    {{0.5, 0.5, 0}, {0, 0, 1}, {0, 1}},
    {{-0.5, 0.5, 0}, {1, 1, 1}, {1, 1}},
    {{-0.5, -0.5, -0.5}, {1, 0, 0}, {1, 0}},
    {{0.5, -0.5, -0.5}, {0, 1, 0}, {0, 0}},
    {{0.5, 0.5, -0.5}, {0, 0, 1}, {0, 1}},
    {{-0.5, 0.5, -0.5}, {1, 1, 1}, {1, 1}}};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class HelloVulkan
{
  public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    GLFWwindow *window;

    vk::UniqueInstance instance;
    vk::DispatchLoaderDynamic dldy;
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debugMessenger;
    vk::UniqueSurfaceKHR surface;

    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;

    vk::UniqueSwapchainKHR swapChain;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::UniqueImageView> swapChainImageViews;
    std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;

    vk::UniqueRenderPass renderPass;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    std::vector<vk::UniquePipeline> graphicsPipelines;

    vk::UniqueCommandPool commandPool;

    vk::UniqueDeviceMemory depthImageMemory;
    vk::UniqueDeviceMemory textureImageMemory;
    vk::UniqueDeviceMemory vertexBufferMemory;
    vk::UniqueDeviceMemory indexBufferMemory;

    vk::UniqueImage depthImage;
    vk::UniqueImage textureImage;
    vk::UniqueBuffer vertexBuffer;
    vk::UniqueBuffer indexBuffer;

    vk::UniqueImageView depthImageView;
    vk::UniqueImageView textureImageView;
    vk::UniqueSampler textureSampler;

    std::vector<vk::UniqueDeviceMemory> uniformBuffersMemory;
    std::vector<vk::UniqueBuffer> uniformBuffers;

    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::CommandBuffer> commandBuffers;

    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::UniqueFence> inFlightFences;
    size_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
    {
        reinterpret_cast<HelloVulkan *>(glfwGetWindowUserPointer(window))->framebufferResized = true;
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
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
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
        while (!glfwWindowShouldClose(window)) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
            }
            glfwPollEvents();
            drawFrame();
        }

        device->waitIdle();
    }

    void cleanup()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void recreateSwapChain()
    {
        int width = 0, height = 0;
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device->waitIdle();
        device->freeCommandBuffers(*commandPool, commandBuffers);

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }

    void drawFrame()
    {
        device->waitForFences({*inFlightFences[currentFrame]}, VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t imageIndex;
        auto result = device->acquireNextImageKHR(*swapChain, std::numeric_limits<uint64_t>::max(), *imageAvailableSemaphores[currentFrame], {}, &imageIndex);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        device->resetFences({*inFlightFences[currentFrame]});

        updateUniformBuffer(imageIndex);

        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        // vk::SubmitInfo(waitSemaphoreCount_, pWaitSemaphores_, pWaitDstStageMask_, commandBufferCount_, pCommandBuffers_, signalSemaphoreCount_, pSignalSemaphores_)
        graphicsQueue.submit({{1, &*imageAvailableSemaphores[currentFrame], &waitStage, 1, &commandBuffers[imageIndex], 1, &*renderFinishedSemaphores[currentFrame]}}, *inFlightFences[currentFrame]);

        auto presentInfo = vk::PresentInfoKHR(1, &*renderFinishedSemaphores[currentFrame], 1, &*swapChain, &imageIndex, nullptr);
        result = presentQueue.presentKHR(&presentInfo);

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo = {
            glm::rotate(glm::mat4(1), time * glm::radians(90.0f), glm::vec3(0, 0, 1)),
            glm::lookAt(glm::vec3(2, 2, 2.), glm::vec3(0, 0, 0), glm::vec3(0, 0, 1)),
            glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f)};

        ubo.proj[1][1] *= -1;

        auto data = device->mapMemory(*uniformBuffersMemory[currentImage], 0, sizeof(ubo), {});
        memcpy(data, &ubo, sizeof(ubo));
        device->unmapMemory(*uniformBuffersMemory[currentImage]);
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores[i] = device->createSemaphoreUnique({});
            renderFinishedSemaphores[i] = device->createSemaphoreUnique({});
            inFlightFences[i] = device->createFenceUnique({vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void createCommandBuffers()
    {
        // vk::CommandBufferAllocateInfo(commandPool_, level_, commandBufferCount_)
        commandBuffers = device->allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t)swapChainFramebuffers.size()});

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            // vk::CommandBufferBeginInfo(flags_, pInheritanceInfo_)
            commandBuffers[i].begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse, nullptr});

            std::vector<vk::ClearValue> clearValues = {
                vk::ClearColorValue(std::array{0.0f, 0.0f, 0.0f, 1.0f}),
                vk::ClearDepthStencilValue(1, 0),
            };

            // vk::RenderPassBeginInfo(renderPass_, framebuffer_, renderArea_, clearValueCount_, pClearValues_)
            auto renderPassInfo = vk::RenderPassBeginInfo(*renderPass, *swapChainFramebuffers[i], {{0, 0}, swapChainExtent}, clearValues.size(), clearValues.data());
            commandBuffers[i].beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipelines[0]);
            commandBuffers[i].bindVertexBuffers(0, *vertexBuffer, {0});
            commandBuffers[i].bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);
            commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, descriptorSets[i], {});
            commandBuffers[i].drawIndexed(indices.size(), 1, 0, 0, 0);
            commandBuffers[i].endRenderPass();
            commandBuffers[i].end();
        }
    }

    void createDescriptorSets()
    {
        auto layouts = std::vector<vk::DescriptorSetLayout>(swapChainImages.size(), *descriptorSetLayout);

        // vk::DescriptorSetAllocateInfo(descriptorPool_, descriptorSetCount_, pSetLayouts_)
        descriptorSets = device->allocateDescriptorSets({*descriptorPool, (uint32_t)swapChainImages.size(), layouts.data()});

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            // vk::DescriptorBufferInfo(buffer_, offset_, range_)
            auto bufferInfo = vk::DescriptorBufferInfo(*uniformBuffers[i], 0, sizeof(UniformBufferObject));
            // vk::DescriptorImageInfo(sampler_, imageView_, imageLayout_)
            auto imageInfo = vk::DescriptorImageInfo(*textureSampler, *textureImageView, vk::ImageLayout::eShaderReadOnlyOptimal);

            // vk::WriteDescriptorSet(dstSet_, dstBinding_, dstArrayElement_, descriptorCount_, descriptorType_, pImageInfo_, pBufferInfo_, pTexelBufferView_)
            device->updateDescriptorSets(
                {{descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo, nullptr},
                 {descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo, nullptr, nullptr}},
                {});
        }
    }

    void createDescriptorPool()
    {
        // vk::DescriptorPoolSize(type_, descriptorCount_)
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eUniformBuffer, (uint32_t)swapChainImages.size()},
            {vk::DescriptorType::eCombinedImageSampler, (uint32_t)swapChainImages.size()}};

        // vk::DescriptorPoolCreateInfo(flags_, maxSets_, poolSizeCount_, pPoolSizes_)
        descriptorPool = device->createDescriptorPoolUnique({{}, (uint32_t)swapChainImages.size(), (uint32_t)poolSizes.size(), poolSizes.data()});
    }

    void createVertexBuffer()
    {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        vk::UniqueDeviceMemory stagingBufferMemory;
        vk::UniqueBuffer stagingBuffer;

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        auto data = device->mapMemory(*stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, vertices.data(), bufferSize);
        device->unmapMemory(*stagingBufferMemory);

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);
        copyBuffer(*stagingBuffer, *vertexBuffer, bufferSize);
    }

    void createIndexBuffer()
    {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        vk::UniqueDeviceMemory stagingBufferMemory;
        vk::UniqueBuffer stagingBuffer;

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        auto data = device->mapMemory(*stagingBufferMemory, 0, bufferSize, {});
        memcpy(data, indices.data(), bufferSize);
        device->unmapMemory(*stagingBufferMemory);

        createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);
        copyBuffer(*stagingBuffer, *indexBuffer, bufferSize);
    }

    void createUniformBuffers()
    {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            createUniqueBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }

    void createUniqueBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueBuffer &buffer, vk::UniqueDeviceMemory &bufferMemory)
    {
        // vk::BufferCreateInfo(flags_, size_, usage_, sharingMode_, queueFamilyIndexCount_, pQueueFamilyIndices_)
        buffer = device->createBufferUnique({{}, size, usage, vk::SharingMode::eExclusive, 0, nullptr});

        auto memRequirements = device->getBufferMemoryRequirements(*buffer);
        uint32_t memoryType = findMemoryType(memRequirements.memoryTypeBits, properties);

        // vk::MemoryAllocateInfo(allocationSize_, memoryTypeIndex_)
        bufferMemory = device->allocateMemoryUnique({memRequirements.size, memoryType});
        device->bindBufferMemory(*buffer, *bufferMemory, 0);
    }

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
    {
        auto uniqueCommandBuffers = beginSingleTimeCommands();

        // vk::BufferCopy(srcOffset_, dstOffset_, size_)
        uniqueCommandBuffers[0]->copyBuffer(srcBuffer, dstBuffer, {{0, 0, size}});
        endSingleTimeCommands(uniqueCommandBuffers);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        auto memProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createTextureSampler()
    {
        textureSampler = device->createSamplerUnique(
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
                0,                                // maxLod_
                vk::BorderColor::eIntOpaqueBlack, // borderColor_
                VK_FALSE                          // unnormalizedCoordinates_
            });
    }

    void createTextureImageView()
    {
        textureImageView = createUniqueImageView(*textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
    }

    void createTextureImage()
    {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        vk::DeviceSize imageSize = texWidth * texHeight * STBI_rgb_alpha;
        vk::UniqueDeviceMemory stagingBufferMemory;
        vk::UniqueBuffer stagingBuffer;

        createUniqueBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        auto data = device->mapMemory(*stagingBufferMemory, 0, imageSize, {});
        memcpy(data, pixels, imageSize);
        device->unmapMemory(*stagingBufferMemory);
        stbi_image_free(pixels);

        createUniqueImage(texWidth, texHeight, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);
        transitionImageLayout(*textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(*stagingBuffer, *textureImage, texWidth, texHeight);
        transitionImageLayout(*textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    void createDepthResources()
    {
        vk::Format depthFormat = findDepthFormat();

        createUniqueImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
        depthImageView = createUniqueImageView(*depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
        transitionImageLayout(*depthImage, depthFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }

    vk::Format findDepthFormat()
    {
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
    {

        for (const auto &format : candidates) {
            auto props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        throw std::runtime_error("failed to find supported format!");
    }

    void createUniqueImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueImage &image, vk::UniqueDeviceMemory &imageMemory)
    {
        // vk::ImageCreateInfo(flags_, imageType_, format_, vk::Extent3D(width_, height_, depth_), mipLevels_, arrayLayers_, samples_, tiling_, usage_, sharingMode_, queueFamilyIndexCount_, pQueueFamilyIndices_, initialLayout_);
        image = device->createImageUnique({{}, vk::ImageType::e2D, format, {width, height, 1U}, 1, 1, vk::SampleCountFlagBits::e1, tiling, usage, vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined});

        auto memRequirements = device->getImageMemoryRequirements(*image);
        uint32_t memoryType = findMemoryType(memRequirements.memoryTypeBits, properties);

        // vk::MemoryAllocateInfo(allocationSize_, memoryTypeIndex_)
        imageMemory = device->allocateMemoryUnique({memRequirements.size, memoryType});
        device->bindImageMemory(*image, *imageMemory, 0);
    }

    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
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

        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        // vk::ImageMemoryBarrier(srcAccessMask_, dstAccessMask_, oldLayout_, newLayout_, srcQueueFamilyIndex_, dstQueueFamilyIndex_, image_, vk::ImageSubresourceRange(aspectMask_, baseMipLevel_, levelCount_, baseArrayLayer_, layerCount_))
        uniqueCommandBuffers[0]->pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, {{srcAccessMask, dstAccessMask, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, {aspectMask, 0, 1, 0, 1}}});
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
        auto uniqueCommandBuffers = device->allocateCommandBuffersUnique({*commandPool, vk::CommandBufferLevel::ePrimary, 1});

        // vk::CommandBufferBeginInfo(flags_, pInheritanceInfo_)
        uniqueCommandBuffers[0]->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit, nullptr});
        return uniqueCommandBuffers;
    }

    void endSingleTimeCommands(std::vector<vk::UniqueCommandBuffer> &uniqueCommandBuffers)
    {
        uniqueCommandBuffers[0]->end();

        // vk::SubmitInfo(waitSemaphoreCount_, pWaitSemaphores_, pWaitDstStageMask_, commandBufferCount_, pCommandBuffers_, signalSemaphoreCount_, pSignalSemaphores_)
        graphicsQueue.submit({{0, nullptr, nullptr, 1, &*uniqueCommandBuffers[0], 0, nullptr}}, nullptr);
        graphicsQueue.waitIdle();
    }

    void createCommandPool()
    {
        // vk::CommandPoolCreateInfo(flags_, queueFamilyIndex_)
        commandPool = device->createCommandPoolUnique({{}, findQueueFamilies(physicalDevice).graphicsFamily.value()});
    }

    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array attachments = {*swapChainImageViews[i], *depthImageView};

            // vk::FramebufferCreateInfo(flags_, renderPass_, attachmentCount_, pAttachments_, width_, height_, layers_)
            swapChainFramebuffers[i] = device->createFramebufferUnique({{}, *renderPass, (uint32_t)attachments.size(), attachments.data(), swapChainExtent.width, swapChainExtent.height, 1});
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

        // vk::Viewport(x_, y_, width_, height_, minDepth_, maxDepth_)
        auto viewport = vk::Viewport(0, 0, swapChainExtent.width, swapChainExtent.height, 0, 1);
        // vk::Rect2D(vk::Offset2D(x_, y_), extent_)
        auto scissor = vk::Rect2D({0, 0}, swapChainExtent);
        // vk::PipelineViewportStateCreateInfo(flags_, viewportCount_, pViewports_, scissorCount_, pScissors_)
        auto viewportState = vk::PipelineViewportStateCreateInfo({}, 1, &viewport, 1, &scissor);

        // vk::PipelineRasterizationStateCreateInfo(flags_, depthClampEnable_, rasterizerDiscardEnable_, polygonMode_, cullMode_, frontFace_, depthBiasEnable_, depthBiasConstantFactor_, depthBiasClamp_, depthBiasSlopeFactor_, lineWidth_)
        auto rasterizer = vk::PipelineRasterizationStateCreateInfo({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, VK_FALSE, 0, 0, 0, 1);

        auto multisampling = vk::PipelineMultisampleStateCreateInfo();

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

        // vk::PipelineLayoutCreateInfo(flags_, setLayoutCount_, pSetLayouts_, pushConstantRangeCount_, pPushConstantRanges_)
        pipelineLayout = device->createPipelineLayoutUnique({{}, 1, &*descriptorSetLayout, 0, nullptr});

        graphicsPipelines = device->createGraphicsPipelinesUnique(
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
                nullptr,                       // pDynamicState_
                *pipelineLayout,               // layout_
                *renderPass,                   // renderPass_
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
        return device->createShaderModuleUnique({{}, buffer.size(), reinterpret_cast<const uint32_t *>(buffer.data())});
    }

    void createDescriptorSetLayout()
    {
        // vk::DescriptorSetLayoutBinding(binding_, descriptorType_, descriptorCount_, stageFlags_, pImmutableSamplers_)
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr},
            {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr}};

        // vk::DescriptorSetLayoutCreateInfo(flags_, bindingCount_, pBindings_)
        descriptorSetLayout = device->createDescriptorSetLayoutUnique({{}, (uint32_t)bindings.size(), bindings.data()});
    }

    void createRenderPass()
    {
        // vk::AttachmentDescription(flags_, format_, samples_, loadOp_, storeOp_, stencilLoadOp_, stencilStoreOp_, initialLayout_, finalLayout_)
        std::vector<vk::AttachmentDescription> attachments = {
            /* colorAttachment*/ {{}, swapChainImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR},
            /* depthAttachment*/ {{}, findDepthFormat(), vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal}};

        // vk::AttachmentReference(attachment_, layout_)
        std::vector<vk::AttachmentReference> attachmentRefs = {
            {0, vk::ImageLayout::eColorAttachmentOptimal},
            {1, vk::ImageLayout::eDepthStencilAttachmentOptimal}};

        // vk::SubpassDescription(flags_, pipelineBindPoint_, inputAttachmentCount_, pInputAttachments_, colorAttachmentCount_, pColorAttachments_, pResolveAttachments_, pDepthStencilAttachment_, preserveAttachmentCount_, pPreserveAttachments_)
        auto subpass = vk::SubpassDescription({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &attachmentRefs[0], nullptr, &attachmentRefs[1], 0, nullptr);

        // vk::SubpassDependency(srcSubpass_, dstSubpass_, srcStageMask_, dstStageMask_, srcAccessMask_, dstAccessMask_, dependencyFlags_)
        auto dependency = vk::SubpassDependency(
            VK_SUBPASS_EXTERNAL, 0,
            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {}, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

        // vk::RenderPassCreateInfo(flags_, attachmentCount_, pAttachments_, subpassCount_, pSubpasses_, dependencyCount_, pDependencies_)
        renderPass = device->createRenderPassUnique({{}, (uint32_t)attachments.size(), attachments.data(), 1, &subpass, 1, &dependency});
    }

    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createUniqueImageView(swapChainImages[i], swapChainImageFormat, vk::ImageAspectFlagBits::eColor);
        }
    }

    vk::UniqueImageView createUniqueImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags)
    {
        // vk::ImageViewCreateInfo(flags_, image_, viewType_, format_, components_, vk::ImageSubresourceRange(aspectMask_, baseMipLevel_, levelCount_, baseArrayLayer_, layerCount_))
        return device->createImageViewUnique({{}, image, vk::ImageViewType::e2D, format, {}, {aspectFlags, 0, 1, 0, 1}});
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

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

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        if (indices.graphicsFamily != indices.presentFamily) {
            imageSharingMode = vk::SharingMode::eConcurrent;
            queueFamilyIndexCount = 2;
            pQueueFamilyIndices = queueFamilyIndices;
        }

        swapChain = device->createSwapchainKHRUnique(
            {
                {},                                             // flags_
                *surface,                                       // surface_
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

        swapChainImages = device->getSwapchainImagesKHR(*swapChain);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
            return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
        }

        for (const auto &availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
    {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;
        for (const auto &availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            } else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = availablePresentMode;
            }
        }
        return bestMode;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            return {std::clamp((uint32_t)width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                    std::clamp((uint32_t)height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
        }
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            // vk::DeviceQueueCreateInfo(flags_, queueFamilyIndex_, queueCount_, pQueuePriorities_)
            queueCreateInfos.push_back(vk::DeviceQueueCreateInfo({}, queueFamily, 1, &queuePriority));
        }

        vk::PhysicalDeviceFeatures deviceFeatures = vk::PhysicalDeviceFeatures().setSamplerAnisotropy(VK_TRUE);

        // vk::DeviceCreateInfo(flags_, queueCreateInfoCount_, pQueueCreateInfos_, enabledLayerCount_, ppEnabledLayerNames, enabledExtensionCount_, ppEnabledExtensionNames_, pEnabledFeatures_)
        device = physicalDevice.createDeviceUnique({{}, (uint32_t)queueCreateInfos.size(), queueCreateInfos.data(), 0, nullptr, (uint32_t)deviceExtensions.size(), deviceExtensions.data(), &deviceFeatures});

        graphicsQueue = device->getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device->getQueue(indices.presentFamily.value(), 0);
    }

    void pickPhysicalDevice()
    {
        for (const auto &device : instance->enumeratePhysicalDevices()) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                return;
            }
        }
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    bool isDeviceSuitable(vk::PhysicalDevice device)
    {
        if (findQueueFamilies(device).isComplete() && checkDeviceExtensionSupport(device)) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            return !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty() && device.getFeatures().samplerAnisotropy;
        }
        return false;
    }

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device)
    {
        return {device.getSurfaceCapabilitiesKHR(*surface), device.getSurfaceFormatsKHR(*surface), device.getSurfacePresentModesKHR(*surface)};
    }

    bool checkDeviceExtensionSupport(vk::PhysicalDevice device)
    {
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto &extension : device.enumerateDeviceExtensionProperties()) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device)
    {
        QueueFamilyIndices indices;

        int i = 0;
        for (const auto &queueFamily : device.getQueueFamilyProperties()) {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            device.getSurfaceSupportKHR(i, *surface, &presentSupport);

            if (queueFamily.queueCount > 0 && presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }
            i++;
        }
        return indices;
    }

    void createSurface()
    {
        VkSurfaceKHR surfaceTmp;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &surfaceTmp) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        // Set the instance as the allocator for the surface for correct order of destruction
        surface = vk::UniqueSurfaceKHR(surfaceTmp, *instance);
    }

    void setupDebugMessenger()
    {
        if (enableValidationLayers) {
            // vk::DebugUtilsMessengerCreateInfoEXT(flags_, messageSeverity_, messageType_, pfnUserCallback_)
            debugMessenger = instance->createDebugUtilsMessengerEXTUnique(
                {{},
                 vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                 vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                 debugCallback},
                nullptr,
                dldy);
        }
    }

    void createInstance()
    {
        // vk::ApplicationInfo(pApplicationName_, applicationVersion_, pEngineName_, engineVersion_, apiVersion_)
        auto appInfo = vk::ApplicationInfo("Hello Vulkan", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_1);

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
        instance = vk::createInstanceUnique({{}, &appInfo, enabledLayerCount, ppEnabledLayerNames, (uint32_t)extensions.size(), extensions.data()});
        dldy.init(*instance);
    }

    bool checkValidationLayerSupport()
    {
        std::set<std::string> requiredLayers(validationLayers.begin(), validationLayers.end());
        for (const auto &layerProperties : vk::enumerateInstanceLayerProperties()) {
            requiredLayers.erase(layerProperties.layerName);
        }
        return requiredLayers.empty();
    }

    std::vector<const char *> getRequiredExtensions()
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

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
    {
        std::cerr << "validation layer: (severity: 0x" << std::setfill('0') << std::setw(4) << std::hex << messageSeverity << ") " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

int main()
{
    HelloVulkan app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
