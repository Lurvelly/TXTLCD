#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <filesystem>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

class ImgLidarAligner{
public:
    struct Pt {
        cv::Point point;
        float dist;
        float z;
        float intensity;
    };
    // ImgLidarAligner();
    void initialize(cv::Mat K1, cv::Mat D1, cv::Mat R1, cv::Mat T1);
    void reset();
    void load(const cv::Mat& img, const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    void getCloudAroundPix(int u, int v, pcl::PointCloud<pcl::PointXYZI>& cloud, cv::Mat& vis);
    void getCloudInPolygon(const std::vector<cv::Point>& polygon, pcl::PointCloud<pcl::PointXYZI>& cloud, pcl::PointCloud<pcl::PointXYZI>& expand_cloud, cv::Mat& vis);
private:
    cv::Mat img_;// suppose img is undistorted
    cv::Mat cloud_;
    pcl::PointCloud<pcl::PointXYZI> original_cloud_;
//    pcl::PointCloud<pcl::PointXYZI> cloud_voi;
    std::vector<float> intensitys_;
    cv::Mat K_, D_, R_, T_;
    std::vector<std::vector<int>> filter_pts_;
    std::vector<std::vector<int>> id_pts_;
    int width_, height_;
    std::vector<Pt> points_;
    void project();
    cv::Scalar fakeColor(float value);
};

// ImgLidarAligner::ImgLidarAligner() {

// }


void ImgLidarAligner::initialize(cv::Mat K1, cv::Mat D1, cv::Mat R1, cv::Mat T1) {
    K_ = K1;
    D_ = D1;
    R_ = R1;
    T_ = T1;
}

void ImgLidarAligner::reset(){
  img_.release();
  cloud_.release();
  original_cloud_.clear();
  intensitys_.clear();
  filter_pts_.clear();
  id_pts_.clear();
  width_ = height_ = 0;
  points_.clear();
}

void ImgLidarAligner::load(const cv::Mat &img, const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
//    cv::Mat I = cv::Mat::eye(3, 3, CV_32FC1);
//    cv::Mat mapX, mapY;
//    cv::initUndistortRectifyMap(K_, D_, I, K_, img.size(), CV_32FC1, mapX, mapY);
//    cv::remap(img, img_, mapX, mapY, cv::INTER_LINEAR);
    img_ = img;//suppose undistortion is done, as text recognision need this too
    width_ = img.cols;
    height_ = img.rows;
    // std::cout << "Image size: " << width_  << ", " << height_ << std::endl;
    // std::cout << "Points size: " << cloud->points.size() << std::endl;
    original_cloud_ = *cloud;
    cloud_ = cv::Mat(cv::Size(cloud->points.size(), 3), CV_32FC1);//(width, height)
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud_.at<float>(0, i) = cloud->points[i].x;//row==0, col==i
        cloud_.at<float>(1, i) = cloud->points[i].y;
        cloud_.at<float>(2, i) = cloud->points[i].z;
        intensitys_.push_back(cloud->points[i].intensity);
    }
    project();
}

void ImgLidarAligner::project() {
//    LOG(INFO) << cloud_.type();
    cv::Mat dist = cloud_.rowRange(0, 1).mul(cloud_.rowRange(0, 1)) +
            cloud_.rowRange(1, 2).mul(cloud_.rowRange(1, 2)) +
            cloud_.rowRange(2, 3).mul(cloud_.rowRange(2, 3));//todo remove this
//    LOG(INFO) << "HERE";

    cv::Mat projCloud2d = K_ * (R_ * cloud_ + repeat(T_, 1, cloud_.cols));
//    float maxDist = 0;
//    float maxIntensity = 0;
//    std::vector<Pt> points;
//    std::vector<std::vector<int>> filter_pts(img_.rows,
//                                             std::vector<int>(img_.cols, -1));
    filter_pts_.resize(img_.rows, std::vector<int>(img_.cols, -1));
    id_pts_.resize(img_.rows, std::vector<int>(img_.cols, -1));
    points_.clear();
//    cloud_voi.clear();
    for (int32_t i = 0; i < projCloud2d.cols; ++i) {
        float x = projCloud2d.at<float>(0, i);
        float y = projCloud2d.at<float>(1, i);
        float z = projCloud2d.at<float>(2, i);
        int x2d = cvRound(x / z);
        int y2d = cvRound(y / z);
        float d = sqrt(dist.at<float>(0, i));
//        float d = 0.0;
        float intensity = intensitys_[i];

        if (x2d >= 0 && y2d >= 0 && x2d < img_.cols && y2d < img_.rows && z > 0) {
//            maxDist = std::max(maxDist, d);
//            maxIntensity = std::max(maxIntensity, intensity);
            points_.push_back(Pt{cv::Point(x2d, y2d), d, z, intensity});
            // add size
            if (filter_pts_[y2d][x2d] != -1) {
                int32_t p_idx = filter_pts_[y2d][x2d];
                if (z < points_[p_idx].z){ //! 保存当前深度更小的这个点
                    filter_pts_[y2d][x2d] = points_.size() - 1;
                    id_pts_[y2d][x2d] = i;
                }
            } else{
                filter_pts_[y2d][x2d] = points_.size() - 1;// id in voi
                id_pts_[y2d][x2d] = i;// id in original
            }
        }
    }
}

void ImgLidarAligner::getCloudInPolygon(const std::vector<cv::Point>& polygon, pcl::PointCloud<pcl::PointXYZI>& cloud, pcl::PointCloud<pcl::PointXYZI>& expand_cloud, cv::Mat& vis){
  cloud.clear();
  expand_cloud.clear();
  // Create a blank image
  // cv::Mat image = cv::imread("/home/jin/Pictures/result/output/1700750607.68.png");

  // // Define the polygon (in this example, a rectangle)
  // std::vector<cv::Point> polygon;
  // polygon.push_back(cv::Point(289.0, 115.0));
  // polygon.push_back(cv::Point(318.0, 116.0));
  // polygon.push_back(cv::Point(319.0, 125.0));
  // polygon.push_back(cv::Point(290.0, 125.0));

  // Fill the polygon with white color to create a mask
  cv::Mat mask = cv::Mat::zeros(img_.size(), CV_8UC1);
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>(1, polygon), cv::Scalar(255));

  int nonZeroCount = cv::countNonZero(mask);
  // std::cout << "nonzero: " << nonZeroCount << std::endl;

  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
  cv::dilate(mask, mask, element);
  nonZeroCount = cv::countNonZero(mask);
  // std::cout << "nonzero: " << nonZeroCount << std::endl;

  // Apply the mask to the original image
  // cv::Mat result;
  img_.copyTo(vis, mask);
  for(int row = 0; row < height_; ++row){
    for(int col = 0; col < width_; ++col){
      if(static_cast<int>(mask.at<uchar>(row, col)) != 255){
        continue;
      }
      int id = id_pts_[row][col];
      if(id == -1){
          continue;
      }
      cloud.push_back(original_cloud_.points[id]);

      int voi_id = filter_pts_[row][col];
      float intensity = points_[voi_id].intensity;
      cv::Scalar color;
      color = fakeColor(intensity / 255.0);
      circle(vis, points_[voi_id].point, 1, color, -1);
    }
  }

  element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(100, 100));
  cv::dilate(mask, mask, element);
  nonZeroCount = cv::countNonZero(mask);
  // std::cout << "nonzero: " << nonZeroCount << std::endl;
  for(int row = 0; row < height_; ++row){
    for(int col = 0; col < width_; ++col){
      if(static_cast<int>(mask.at<uchar>(row, col)) != 255){
        continue;
      }
      int id = id_pts_[row][col];
      if(id == -1){
          continue;
      }
      expand_cloud.push_back(original_cloud_.points[id]);
    }
  }

}


void ImgLidarAligner::getCloudAroundPix(int u, int v, pcl::PointCloud<pcl::PointXYZI>& cloud, cv::Mat& vis){
    cloud.clear();
    int neighbor = 30;//3
    for(int row = std::max(0, v - neighbor); row < std::min(v + neighbor, height_); row++){
        for(int col = std::max(0, u - neighbor); col < std::min(u + neighbor, width_); col++){
//            int id = filter_pts_[row][col];
//            if(id == -1){
//                continue;
//            }
//            pcl::PointXYZI p;
//            p.x = points_[id].point.x * points_[id].z;
//            p.y = points_[id].point.y * points_[id].z;
//            p.z = points_[id].z;
//            p.intensity = points_[id].intensity;
//            cloud.push_back(p);
            int id = id_pts_[row][col];
            if(id == -1){
                continue;
            }
            cloud.push_back(original_cloud_.points[id]);

            int voi_id = filter_pts_[row][col];
            float intensity = points_[voi_id].intensity;
            cv::Scalar color;
            color = fakeColor(intensity / 255.0);
            circle(vis, points_[voi_id].point, 1, color, -1);
        }
    }
}

cv::Scalar ImgLidarAligner::fakeColor(float value) {
    float posSlope = 255 / 60.0;
    float negSlope = -255 / 60.0;
    value *= 255;
    cv::Vec3f color;
    if (value < 60) {
        color[0] = 255;
        color[1] = posSlope * value + 0;
        color[2] = 0;
    } else if (value < 120) {
        color[0] = negSlope * value + 2 * 255;
        color[1] = 255;
        color[2] = 0;
    } else if (value < 180) {
        color[0] = 0;
        color[1] = 255;
        color[2] = posSlope * value - 2 * 255;
    } else if (value < 240) {
        color[0] = 0;
        color[1] = negSlope * value + 4 * 255;
        color[2] = 255;
    } else if (value < 300) {
        color[0] = posSlope * value - 4 * 255;
        color[1] = 0;
        color[2] = 255;
    } else {
        color[0] = 255;
        color[1] = 0;
        color[2] = negSlope * value + 6 * 255;
    }
    return cv::Scalar(color[0], color[1], color[2]);
}

// ImgLidarAligner aligner;

class TxtObject{
public:
  typedef std::shared_ptr<TxtObject> Ptr;
  TxtObject(const std::string& content, int frame_id, int object_id, const Eigen::Vector3d& center, const Eigen::Vector3d& norm){
    content_ = content;
    frame_id_ = frame_id;
    object_id_ = object_id;
    center_ = center;
    norm_ = norm;
  }
  TxtObject(const std::string& content, int frame_id, int object_id){
    content_ = content;
    frame_id_ = frame_id;
    object_id_ = object_id;
    
    // center_ = center;
    // norm_ = norm;
  }

  bool isValid() const {
    return type_ != type::INVALID;
  }

  std::string content_;
  int frame_id_;// 第几帧识别到的
  int object_id_;// 这一帧中的第几个object
  int global_id_;// todo 全局object的编号
  int global_class_id_;// todo 全局同类型object的编号
  
  enum type {ORDINARY = 0, ROOMNUM = 1, EXIT = 2, INVALID = 3};
  type type_ = type::ORDINARY;//

  Eigen::Vector3d center_;// in frame of frame_id
  Eigen::Vector3d norm_;
  Eigen::Vector3d main_direction_;
  int u1_, v1_, u2_, v2_;
  std::vector<cv::Point> polygon_;
};

// vector<pair<TxtObject::Ptr, TxtObject::Ptr>> loopObjectsQueue;

class Frame{
public:
  typedef std::shared_ptr<Frame> Ptr;
  Frame(int index){
    frame_id_ = index;//todo initialize pose
  }
  int frame_id_;
  
  std::string lidar_timestamp_tag_;
  // Eigen::Quaterniond rotation_;
  // Eigen::Vector3d position_;
  std::vector<TxtObject::Ptr> objects_;
  bool valid_ = true;
};

class TxtSet{// save txt of the same string
public:
  typedef std::shared_ptr<TxtSet> Ptr;
  TxtSet(const std::string& content){
    content_ = content;
    objects_.clear();
  }
  void add(const TxtObject::Ptr object){//make sure (frame_id, object_id) paired are not added repetitively
    objects_.insert(std::make_pair(object->frame_id_, object->object_id_));
  }

  // void retrieve(const TxtObject::Ptr object){//make sure (frame_id, object_id) paired are not added repetitively
  //   Eigen::Vector source_pose = object->
  // }
public:
  std::string content_;
  std::map<int, int> objects_;// first is frame id, second is object id
};

class TxtManager{
public:
  typedef std::shared_ptr<TxtManager> Ptr;
  TxtManager(){
    frames_.clear();
    object_sets_.clear();
    std::cout << "Object set size: " << object_sets_.size() << std::endl; 
  }
  
  void add(Frame::Ptr new_frame){
    frames_.push_back(new_frame);
    // for(const TxtObject::Ptr object : new_frame->objects_){
    //   //todo type
    //   std::string content = object->content_;
    //   if(object_sets_.find(content) == object_sets_.end()){
    //     TxtSet::Ptr txt_set(new TxtSet(content));
    //     txt_set->add(object);
    //     object_sets_.insert(std::make_pair(content, txt_set));
    //   }else{
    //     TxtSet::Ptr current_set = object_sets_[content];
    //     current_set->add(object);
    //   }
    // }
  }

  void refreshFrame(int frame_id){
    const Frame::Ptr& current_frame = frames_.at(frame_id);
    for(const TxtObject::Ptr object : current_frame->objects_){
      //todo type
      if(object->type_ == TxtObject::type::INVALID){
        continue;
      }
      std::string content = object->content_;
      if(object_sets_.find(content) == object_sets_.end()){
        TxtSet::Ptr txt_set(new TxtSet(content));
        txt_set->add(object);
        object_sets_.insert(std::make_pair(content, txt_set));
      }else{
        TxtSet::Ptr current_set = object_sets_[content];
        current_set->add(object);
      }
    }
  }

  // void updateFramePose(int index, Eigen::Vector3d position, )

  bool retrieveObject(TxtObject::Ptr source_object, std::vector<TxtObject::Ptr>& loop){
    loop.clear();
    std::string content = source_object->content_;
    // sleep(10);
    // std::cout << "###############################retrieve content: " << content << std::endl;
    // std::cout << &object_sets_ << std::endl;
    // std::cout << object_sets_.size() << std::endl;
    if(object_sets_.empty()){
      std::cout << "empty, retur here" << std::endl;
      return false;
    }
    if(object_sets_.find(content) == object_sets_.end()){
      return false;
    }
    // Frame::Ptr source_frame = frame[source_object->frame_id_];//todo make sure the frame_id is equal with the index in vector, as well as key fram id
    // Eigen::Vector3d source_position = source_frame->rotation.matrix() * source_object->center_ + source_frame->position_;// todo if its enough
    TxtSet::Ptr current_set = object_sets_[content];//double search
    for(const std::pair<int, int>& object : current_set->objects_){
      // Frame::Ptr current_frame = frame[object->frame_id_];
      // Eigen::Vector3d current_position = current_frame->rotation.matrix() * object->center_ + current_frame->position_;
      // if((current_position - source_postion).norm() < 0.5){// todo this needs to be modified while debug, and for unique txt object its not true
        if(object.first == source_object->frame_id_) continue;//! retrieve的时候，本帧已经在manager里

        loop.push_back(frames_[object.first]->objects_[object.second]);
      // }
    }
    if(!loop.empty()){
      std::cout << "============Find object set with same content for txt: " << content << std::endl;
      // for(const auto& o : loop){
      //   std::cout << source_object->frame_id_ << " vs " << o->frame_id_ << std::endl;
      // }      
    }
    // std::cout << source_object->frame_id_ << " vs " << object_sets_[content]->objects_.begin()->first << std::endl;
    return !loop.empty();
  }

  int getFrameSize() const {return frames_.size();}

  Frame::Ptr getFrame(int index) const {return frames_.at(index);}

  // Eigen::Vector3d getObjectPose(const std::vector) 
public:
  std::vector<Frame::Ptr, std::allocator<Frame::Ptr>> frames_;
  std::unordered_map<std::string, TxtSet::Ptr> object_sets_;
};

