#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <optional>
#include <string>


// ===================== 配置结构体定义 =====================
struct InitModelConfig {
    std::string alias;          // 别名
    int class_cnt;              // 类别数量
    int model_type;             // 模型类型
    int model_size[2];          // 模型尺寸 [宽, 高]
    bool is_cuda;               // 是否使用CUDA
    int cuda_id;                // CUDA设备ID
    std::string mode_path;      // 模型路径
    std::string classes_path;   // 类别文件路径
    float score_thres;          // 置信度阈值
    float nms_thres;            // NMS阈值
};

struct InferenceConfig {
    bool is_crop;               // 是否裁剪
    struct Roi {                // ROI区域
        int x;
        int y;
        int w;
        int h;
    } roi;
    bool save_result;           // 是否保存结果
    std::string result_path;    // 结果保存路径
};

// ===================== 核心解析函数声明 =====================
/**
 * @brief 读取JSON配置文件并解析到两个配置结构体
 * @param json_path JSON文件路径
 * @param init_model 输出：init_model配置
 * @param inference 输出：inference配置
 * @return 错误信息（无错误返回std::nullopt）
 */
std::optional<std::string> readConfig(const std::string& json_path, 
                                     InitModelConfig& init_model, 
                                     InferenceConfig& inference);

/**
 * @brief 打印配置信息（调试用）
 * @param init_model init_model配置
 * @param inference inference配置
 */
void printConfig(const InitModelConfig& init_model, const InferenceConfig& inference);

#endif // CONFIG_PARSER_H