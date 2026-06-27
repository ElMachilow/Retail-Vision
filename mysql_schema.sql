CREATE DATABASE IF NOT EXISTS `retail_vision`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `retail_vision`;

CREATE TABLE IF NOT EXISTS `users` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `username` VARCHAR(80) NOT NULL UNIQUE,
  `password_hash` VARCHAR(128) NOT NULL,
  `password_salt` VARCHAR(64) NOT NULL,
  `display_name` VARCHAR(120),
  `role` VARCHAR(40) NOT NULL DEFAULT 'user',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX `idx_users_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `users` (`username`, `password_hash`, `password_salt`, `display_name`, `role`) VALUES (
  'admin',
  '582eba7cc210555498746cfc032e628730db8ac72d4d8fec5cd9eb62d878522d',
  '9fbe408d9dcd54b04d182f98ed35db13',
  'Administrador',
  'admin'
);

CREATE TABLE IF NOT EXISTS `productos` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `nombre_producto` VARCHAR(255) NOT NULL,
  `marca` VARCHAR(120),
  `tipo_producto` VARCHAR(120),
  `presentacion` VARCHAR(120),
  `contenido_neto` VARCHAR(60),
  `unidad_medida` VARCHAR(20),
  `categoria_sugerida` VARCHAR(120),
  `codigo_barras` VARCHAR(64) UNIQUE,
  `precio_venta` DECIMAL(10, 2) NOT NULL DEFAULT 0,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX `idx_productos_nombre` (`nombre_producto`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `recognition_events` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `trace_id` VARCHAR(120) NOT NULL,
  `source_name` VARCHAR(255),
  `image_content_type` VARCHAR(120),
  `image_blob` LONGBLOB,
  `image_path` VARCHAR(500),
  `status` VARCHAR(40) NOT NULL DEFAULT 'pending_review',
  `predicted_nombre_producto` VARCHAR(255),
  `predicted_marca` VARCHAR(120),
  `predicted_tipo_producto` VARCHAR(120),
  `predicted_presentacion` VARCHAR(120),
  `predicted_contenido_neto` VARCHAR(60),
  `predicted_unidad_medida` VARCHAR(20),
  `predicted_categoria_sugerida` VARCHAR(120),
  `final_nombre_producto` VARCHAR(255),
  `final_marca` VARCHAR(120),
  `final_tipo_producto` VARCHAR(120),
  `final_presentacion` VARCHAR(120),
  `final_contenido_neto` VARCHAR(60),
  `final_unidad_medida` VARCHAR(20),
  `final_categoria_sugerida` VARCHAR(120),
  `final_codigo_barras` VARCHAR(64),
  `yolo_confidence` DOUBLE,
  `yolo_class_name` VARCHAR(120),
  `ocr_confidence` DOUBLE,
  `ocr_text` TEXT,
  `warnings_json` JSON NOT NULL DEFAULT (JSON_ARRAY()),
  `bbox_json` JSON,
  `failure_reason` VARCHAR(120),
  `review_notes` VARCHAR(500),
  `reviewed_by_user_id` INT,
  `reviewed_by_username` VARCHAR(120),
  `use_for_training` TINYINT(1) NOT NULL DEFAULT 0,
  `linked_product_id` INT,
  `recognition_json` JSON NOT NULL,
  `reviewed_at` DATETIME,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX `idx_recognition_events_status` (`status`),
  INDEX `idx_recognition_events_created_at` (`created_at`),
  CONSTRAINT `fk_recognition_product` FOREIGN KEY (`linked_product_id`) REFERENCES `productos`(`id`) ON DELETE SET NULL,
  CONSTRAINT `fk_recognition_user` FOREIGN KEY (`reviewed_by_user_id`) REFERENCES `users`(`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `inventory_sessions` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `nombre` VARCHAR(120) NOT NULL,
  `estado` VARCHAR(40) NOT NULL DEFAULT 'open',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `closed_at` DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `inventory_items` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `session_id` INT NOT NULL,
  `product_id` INT,
  `recognition_event_id` INT,
  `nombre_producto` VARCHAR(255) NOT NULL,
  `marca` VARCHAR(120),
  `tipo_producto` VARCHAR(120),
  `categoria` VARCHAR(120),
  `contenido_neto` VARCHAR(60),
  `unidad_medida` VARCHAR(20),
  `cantidad` INT NOT NULL DEFAULT 1,
  `ubicacion` VARCHAR(120),
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX `idx_inventory_items_session` (`session_id`),
  INDEX `idx_inventory_items_category` (`categoria`),
  CONSTRAINT `fk_inventory_session` FOREIGN KEY (`session_id`) REFERENCES `inventory_sessions`(`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_inventory_product` FOREIGN KEY (`product_id`) REFERENCES `productos`(`id`) ON DELETE SET NULL,
  CONSTRAINT `fk_inventory_recognition` FOREIGN KEY (`recognition_event_id`) REFERENCES `recognition_events`(`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `product_stock_counts` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `mobile_product_id` VARCHAR(120),
  `nombre_producto` VARCHAR(255) NOT NULL,
  `cantidad_final` INT NOT NULL,
  `confianza` DOUBLE NOT NULL DEFAULT 0,
  `total_fotos` INT NOT NULL DEFAULT 0,
  `valid_fotos` INT NOT NULL DEFAULT 0,
  `source` VARCHAR(40) NOT NULL DEFAULT 'mobile',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_product_stock_counts_mobile_product` (`mobile_product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `product_stock_count_photos` (
  `id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `count_id` INT NOT NULL,
  `recognition_event_id` INT,
  `source_name` VARCHAR(255),
  `detected_name` VARCHAR(255),
  `matched` TINYINT(1) NOT NULL DEFAULT 0,
  `accepted` TINYINT(1) NOT NULL DEFAULT 0,
  `confidence` DOUBLE NOT NULL DEFAULT 0,
  `warnings_json` JSON NOT NULL DEFAULT (JSON_ARRAY()),
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_product_stock_count_photos_count` (`count_id`),
  CONSTRAINT `fk_stock_photo_count` FOREIGN KEY (`count_id`) REFERENCES `product_stock_counts`(`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_stock_photo_recognition` FOREIGN KEY (`recognition_event_id`) REFERENCES `recognition_events`(`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'visionai_app'@'localhost' IDENTIFIED BY 'VisionAI@2026';
GRANT ALL PRIVILEGES ON `retail_vision`.* TO 'visionai_app'@'localhost';
FLUSH PRIVILEGES;
